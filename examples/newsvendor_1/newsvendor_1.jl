import GLM
import Flux
import Random
import HiGHS
import BilevelJuMP
import Distributions
using Plots
using JuMP
using ApplicationDrivenLearning

Random.seed!(0)

# entry parameters
T = 30  # timesteps
p = 1  # AR-p
I = 1  # newsvendors
μ = 100  # demand mean
σ = 10  # demand variance
α = 1.03  # AR model parameter
c = [5]  # nesvendor cost
q = [9]  # newsvendor price
r = [4]  # newsvendor salvage value

IMGS_PATH = joinpath(@__DIR__, "imgs")
if !isdir(IMGS_PATH)
    mkdir(IMGS_PATH)
end

# generate data X,Y
Y = Matrix{Float32}(undef, T+p, I)
Y[1:p, :] .= rand(Distributions.Normal(μ, σ), (p, I))
for t in (p+1):T+p
    Y[t, :] = α .* Y[t-1, :] .+ rand(Distributions.Normal(0.0, σ), I)
end

X = Y[1:end-p, :]
Y = Y[p+1:end, :]

# compute least-squares model
linear_ls_model = GLM.lm(hcat(X, ones(T)), Float64.(Y[:, 1]))
θ = Float32.(GLM.coef(linear_ls_model))

# build application driven learning model
model = ApplicationDrivenLearning.Model()
@variables(model, begin
    x[1:I] ≥ 0, ApplicationDrivenLearning.Policy
    d[1:I] ≥ 0, ApplicationDrivenLearning.Forecast
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
set_optimizer(model, HiGHS.Optimizer)
set_silent(model)

# declare predictive model
nn = Flux.Chain(
    Flux.Dense(p => 1)
)
Flux.params(nn)[1][1, :] .= θ[1:end-1]
Flux.params(nn)[2][1] = θ[end]
ApplicationDrivenLearning.set_forecast_model(model, nn)

# least-squares model prediction and cost
yhat_ls = model.forecast(X')'
cost_ls = ApplicationDrivenLearning.compute_cost(model, X, Y)

# train with bilevel mode
bl_sol = ApplicationDrivenLearning.train!(
    model, X, Y,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.BilevelMode,
        optimizer=HiGHS.Optimizer,
        mode=BilevelJuMP.FortunyAmatMcCarlMode(primal_big_M=5000, dual_big_M=5000)
    )
)
yhat_opt = model.forecast(X')'
cost_opt = ApplicationDrivenLearning.compute_cost(model, X, Y)

# compare predictions
fig1 = plot(Y, label="True", color=:grey, title="Newsvendor Demand", xlabel="Timesteps")
plot!(yhat_ls, label="LS")
plot!(yhat_opt, label="Opt")
savefig(fig1, joinpath(IMGS_PATH, "demand.png"))

# compare errors
fig2 = plot((yhat_ls .- Y).^2, label="LS (Cost=$(round(cost_ls, digits=2)))", title="MSE", xlabel="Timesteps")
plot!((yhat_opt .- Y).^2, label="Opt (Cost=$(round(cost_opt, digits=2)))")
savefig(fig2, joinpath(IMGS_PATH, "errors.png"))