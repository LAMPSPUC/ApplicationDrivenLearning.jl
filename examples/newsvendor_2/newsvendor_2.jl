import GLM
import Flux
import Random
import Gurobi
import StatsAPI
import Statistics
import BilevelJuMP
import Distributions
using Plots
using JuMP
using ApplicationDrivenLearning

Random.seed!(0)

# entry parameters
T = 1000  # timesteps
p = 4  # AR-p
I = 2  # newsvendors
μ = 10  # demand mean
σ = 3  # demand variance
c = [10, 10]  # nesvendor cost
q = [19, 11]  # newsvendor price
r = [9, 1]  # newsvendor salvage value

IMGS_PATH = joinpath(@__DIR__, "imgs")
if !isdir(IMGS_PATH)
    mkdir(IMGS_PATH)
end

# generate data X,Y
Y = Matrix{Float32}(undef, T+p, I)
Y[1:p, :] .= rand(Distributions.Normal(μ, σ), (p, I))
α = rand(Distributions.Uniform(0.1, 1.0), (p, I))
α = α ./ sum(α, dims=1)  # Normalize to sum to 1
for t in (p+1):T+p
    for i=1:I
        Y[t, i] = (α[:, i]'Y[t-p:t-1, i])' .+ rand(Distributions.Normal(0.0, σ))
    end
end
X = Matrix{Float32}(undef, T, I*p)
for t=1:T
    for ip=1:p
        for i=1:I
            X[t, p*(i-1) + ip] = Y[t+ip-1, i]
        end
    end
end
Y = Y[p+1:end, :]

# compute least-squares models (one for each item)
θ = zeros((p+1, I))
ls_models = []
for i=1:I
    xs = hcat(X[:, p*(i-1)+1:p*i], ones(T)) .|> Float64
    ys = Y[:, i] .|> Float64
    ls_model = GLM.lm(xs, ys)
    push!(ls_models, ls_model)
    θ[:, i] .= Float32.(GLM.coef(ls_model))
end

# build application driven learning model
model = ApplicationDrivenLearning.Model()
@variables(model, begin
    x[1:I] ≥ 0, ApplicationDrivenLearning.Policy
    d[1:I], ApplicationDrivenLearning.Forecast
end)
function build_nesvendor_jump_model(jump_model, x, d)
    @variables(jump_model, begin
        y[1:I] ≥ 0
        w[1:I] ≥ 0
    end)
    @constraints(jump_model, begin
        con1[i=1:I], y[i] ≤ d[i]
        con2[i=1:I], y[i] + w[i] ≤ x[i]
    end)
    cost_exp = @expression(jump_model, c'x-q'y-r'w)
    @objective(jump_model, Min, cost_exp)
end
build_nesvendor_jump_model(ApplicationDrivenLearning.Plan(model), [i.plan for i in x], [i.plan for i in d])
build_nesvendor_jump_model(ApplicationDrivenLearning.Assess(model), [i.assess for i in x], [i.assess for i in d])
set_optimizer(model, Gurobi.Optimizer)
set_silent(model)

# declare predictive model
nns = [
    Flux.Chain(Flux.Dense(p => 1))
    for i=1:I
]
for i=1:I
    Flux.params(nns[i])[1][1, :] .= θ[1:end-1, i]
    Flux.params(nns[i])[2][1] = θ[end, i]
end
input_output_map = [
    Dict(
        collect((i-1)*p+1:i*p) => [i]
    )
    for i=1:I
]
ApplicationDrivenLearning.set_forecast_model(
    model,
    ApplicationDrivenLearning.PredictiveModel(
        deepcopy(nns), 
        input_output_map, 
        size(X, 2),
        size(Y, 2)
    )
)

# least-squares model prediction and cost
yhat_ls = model.forecast(X')'
cost_ls = ApplicationDrivenLearning.compute_cost(model, X, Y)

# train with bilevel mode
bl_sol = ApplicationDrivenLearning.train!(
    model, X, Y,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.BilevelMode,
        optimizer=Gurobi.Optimizer,
        mode=BilevelJuMP.FortunyAmatMcCarlMode(primal_big_M=5000, dual_big_M=5000)
    )
)
yhat_opt = model.forecast(X')'
cost_opt = ApplicationDrivenLearning.compute_cost(model, X, Y)

println("Cost LS: $cost_ls | Cost OPT: $cost_opt")
println("Improvement: $(100*(cost_opt-cost_ls)/ cost_ls)%")

# uncertainty analysis
bias = Matrix{Float64}(undef, T, I)
for i=1:I
    bias[:, i] .= abs.(yhat_opt[:, i] .- yhat_ls[:, i])
end
uncertainty = Matrix{Float64}(undef, T, I)
for i=1:I
    xs = hcat(X[:, p*(i-1)+1:p*i], ones(T)) .|> Float64
    prediction, lower, upper = StatsAPI.predict(ls_models[i], xs; interval=:confidence, level=0.95)
    uncertainty[:, i] .= upper - lower
end
corr = [Statistics.cor(uncertainty[:, i], bias[:, i]) for i=1:I]

# compare predictions
MAX_TIMESTEPS_PLOT = 100
fig1 = plot(Y[1:MAX_TIMESTEPS_PLOT, 1], label="True", color=:black, title="Newsvendor 1 Demand", xlabel="Timesteps")
plot!(yhat_ls[1:MAX_TIMESTEPS_PLOT, 1], label="LS", color=:grey)
plot!(yhat_opt[1:MAX_TIMESTEPS_PLOT, 1], label="Opt")
savefig(fig1, joinpath(IMGS_PATH, "newsvendor_1_demand.png"))

fig2 = plot(Y[1:MAX_TIMESTEPS_PLOT, 2], label="True", color=:black, title="Newsvendor 2 Demand", xlabel="Timesteps")
plot!(yhat_ls[1:MAX_TIMESTEPS_PLOT, 2], label="LS", color=:grey)
plot!(yhat_opt[1:MAX_TIMESTEPS_PLOT, 2], label="Opt")
savefig(fig2, joinpath(IMGS_PATH, "newsvendor_2_demand.png"))

# compare errors
err_ls = (yhat_ls .- Y)
err_opt = (yhat_opt .- Y)
min_bin = minimum([minimum(err_ls), minimum(err_opt)])
max_bin = maximum([maximum(err_ls), maximum(err_opt)])
bins = min_bin:1:max_bin
fig3 = histogram(
    err_ls[:, 1], 
    alpha=.7,
    bins=bins,
    label="LS", color=:grey,
    title="Newsvendor 1 Error", 
    xlabel="Timesteps"
)
histogram!(
    err_opt[:, 1], 
    alpha=.7,
    bins=bins,
    label="Opt"
)
savefig(fig3, joinpath(IMGS_PATH, "newsvendor_1_errors.png"))

fig4 = histogram(
    err_ls[:, 2], 
    alpha=.7,
    bins=bins,
    label="LS", color=:grey, 
    title="Newsvendor 2 Error", 
    xlabel="Timesteps"
)
histogram!(
    err_opt[:, 2], 
    alpha=.7,
    bins=bins,
    label="Opt"
)
savefig(fig4, joinpath(IMGS_PATH, "newsvendor_2_errors.png"))

# scatter plot
fig5 = plot(
    title="Relationship between uncertainty and bias",
    xlabel="Bias",
    ylabel="Uncertainty"
)
for i=1:I
    scatter!(
        bias[:, i],
        uncertainty[:, i],
        alpha=.7,
        label="Newsvendor $i (corr=$(round(100*corr[i], digits=1))%)",
    )
end
savefig(joinpath(IMGS_PATH, "uncertainty_bias.png"))
