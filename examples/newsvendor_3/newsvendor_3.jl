import Flux
import HiGHS
import Random
import Statistics
import BilevelJuMP
import ColorSchemes
import Distributions
using JuMP
using Plots
using DataFrames
using ApplicationDrivenLearning

include("data.jl")
include("lp.jl")
include("ls.jl")

# paths
IMGS_PATH = joinpath(@__DIR__, "imgs")
if !isdir(IMGS_PATH)
    mkdir(IMGS_PATH)
end

# parameters
T = 100
I = 2
p = 4
p2 = 12

# generate_data
X, Y = generate_series_data(I, T, p2)
X = hcat(X[:, 1:p], X[:, p2+1:p2+p])
# init model
model = init_newsvendor_model(I, HiGHS.Optimizer);

function get_nn(p::Int, hidden_layers::Int, hidden_layers_size::Int=64)
    Random.seed!(0)

    if hidden_layers == 0
        return Flux.Chain(Flux.Dense(p => 1))
    else
        return Flux.Chain(
            Flux.Dense(p => hidden_layers_size, Flux.relu),
            [Flux.Dense(hidden_layers_size, hidden_layers_size, Flux.relu) for _ in 1:hidden_layers-1]...,
            Flux.Dense(hidden_layers_size => 1, Flux.relu)
        )
    end
end

rows = []
n_hidden_layers_space = [0, 1, 2]

for n_hidden_layers in n_hidden_layers_space
    # start model
    ls_nn = get_nn(p, n_hidden_layers)
    layers_size = size.(Flux.params(ls_nn))
    n_params = sum(prod.(layers_size))
    
    # train nn with least-squares
    t0 = time()
    if n_hidden_layers == 0
        ls_θ = get_ls_solution(X, Y)
        Flux.params(ls_nn)[1][1, :] .= value.(ls_θ[1:end-1])
        Flux.params(ls_nn)[2][1] = value(ls_θ[end])
    else
        train_nn(ls_nn, X, Y, Flux.Adam(), 1000, 20)
    end
    time_ls = time() - t0
    
    # set forecast model
    input_output_map = Dict(
        collect((i-1)*p+1:i*p) => [i]
        for i=1:I
    )
    ApplicationDrivenLearning.set_forecast_model(
        model,
        ApplicationDrivenLearning.PredictiveModel(deepcopy(ls_nn), input_output_map)
    )

    # ls prediction
    yhat_ls = model.forecast(X')'
    yerr_ls = Statistics.mean(sum((yhat_ls - Y).^2, dims=2))
    cost_ls = ApplicationDrivenLearning.compute_cost(model, X, Y)

    # train with bilevel mode
    if n_hidden_layers == 0
        t0 = time()
        bl_sol = ApplicationDrivenLearning.train!(
            model, X, Y,
            ApplicationDrivenLearning.Options(
                ApplicationDrivenLearning.BilevelMode,
                optimizer=HiGHS.Optimizer,
                mode=BilevelJuMP.FortunyAmatMcCarlMode(primal_big_M=5000, dual_big_M=5000)
            )
        )
        time_bl = time() - t0
        yhat_bl = model.forecast(X')'
        yerr_bl = Statistics.mean(sum((yhat_bl - Y).^2, dims=2))
        cost_bl = ApplicationDrivenLearning.compute_cost(model, X, Y)
    else
        yerr_bl = NaN
        cost_bl = NaN
        time_bl = NaN
    end

    # train with nelder mead
    if n_hidden_layers <= 2
        ls_nn = Flux.Chain(ls_nn..., Flux.relu)
        ApplicationDrivenLearning.set_forecast_model(
            model,
            ApplicationDrivenLearning.PredictiveModel(
                deepcopy(ls_nn), 
                input_output_map
            )
        )
        t0 = time()
        nm_sol = ApplicationDrivenLearning.train!(
            model, X, Y,
            ApplicationDrivenLearning.Options(
                ApplicationDrivenLearning.NelderMeadMode,
                iterations=30_000, 
                show_trace=true, 
                show_every=100,
                time_limit=60,
            )
        )
        time_nm = time() - t0
        yhat_nm = model.forecast(X')'
        yerr_nm = Statistics.mean(sum((yhat_nm - Y).^2, dims=2))
        cost_nm = ApplicationDrivenLearning.compute_cost(model, X, Y)
    else
        yerr_nm = NaN
        cost_nm = NaN
        time_nm = NaN
    end

    # train with gradient descent
    ApplicationDrivenLearning.set_forecast_model(
        model,
        ApplicationDrivenLearning.PredictiveModel(deepcopy(ls_nn), input_output_map)
    )
    t0 = time()
    gd_sol = ApplicationDrivenLearning.train!(
        model, X, Y,
        ApplicationDrivenLearning.Options(
            ApplicationDrivenLearning.GradientMode,
            rule=Flux.Adam(1e-4),
            epochs=1_000,
            time_limit=60
        )
    )
    time_gd = time() - t0
    yhat_gd = model.forecast(X')'
    yerr_gd = Statistics.mean(sum((yhat_gd - Y).^2, dims=2))
    cost_gd = ApplicationDrivenLearning.compute_cost(model, X, Y)

    push!(rows, (
        n_hidden_layers, 
        n_params, 
        time_ls, 
        yerr_ls, 
        cost_ls, 
        time_bl, 
        yerr_bl, 
        cost_bl, 
        time_nm, 
        yerr_nm, 
        cost_nm, 
        time_gd, 
        yerr_gd, 
        cost_gd
    ))

end


df = DataFrame(
    rows, 
    [
        :n_layers, :n_params,
        :time_ls, :yerr_ls, :cost_ls,
        :time_bl, :yerr_bl, :cost_bl,
        :time_nm, :yerr_nm, :cost_nm,
        :time_gd, :yerr_gd, :cost_gd
    ]
)

# plot
fig = plot()
colors = [
    get(ColorSchemes.rainbow, 0), 
    get(ColorSchemes.rainbow, 0.25), 
    get(ColorSchemes.rainbow, 0.5),
    get(ColorSchemes.rainbow, 0.75),
]
title!("In-Sample Assess Cost by model size")
xlabel!("Number of hidden layers")
plot!(
    df.n_layers, df.cost_ls, label="LS", 
    color=colors[1], alpha=.7, linewidth=2, markersize=5, markershape=:square
)
plot!(
    df.n_layers, df.cost_nm, label="LS+NM", 
    color=colors[3], alpha=.7, linewidth=2, markersize=5, markershape=:diamond
)
plot!(
    df.n_layers, df.cost_gd, label="LS+GD", 
    color=colors[4], alpha=.7, linewidth=2, markersize=5, markershape=:utriangle
)
plot!(
    df.n_layers, df.cost_bl, label="BL", 
    color=:black, linewidth=2, markersize=5, markershape=:circle
)
fig
savefig(fig, joinpath(IMGS_PATH, "modes_compare.png"))
