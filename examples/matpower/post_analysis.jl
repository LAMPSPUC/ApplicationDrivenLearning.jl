# computes train and test costs for different versions of the model

import Parameters
using JuMP
import DelimitedFiles
import Random
using Statistics
import OffsetArrays: OffsetArray
import Gurobi
using Flux
import NNlib
using Plots
import Distributions
import JLD2
using DataFrames
import CSV
using ApplicationDrivenLearning


include("config.jl")
pretrain = false
gradient_mode = false
neldermead_mode = false

include("utils/struct.jl")
include("utils/data.jl")
include("utils/model.jl")
include("utils/pretrain.jl")

gradient_model_state = joinpath(result_path, "model_state_gd.jld2")
neldermead_model_state = joinpath(result_path, "model_state_nm.jld2")

###########################################################################

# get LS model
models_state = JLD2.load(pretrained_model_state, "state")
Flux.loadmodel!(nns, models_state);
pred_model = ADL.PredictiveModel(
    nns, 
    input_output_map, 
    lags*pd.n_demand+1, 
    pd.n_demand+2*pd.n_zones
)
ADL.set_forecast_model(
    model,
    deepcopy(pred_model)
)
ls_pred_train = model.forecast(X_train')'
ls_cost_train = ADL.compute_cost(model, X_train, Y_train, false, false)
ls_pred_test = model.forecast(X_test')'
ls_cost_test = ADL.compute_cost(model, X_test, Y_test, false, false)

# get GD model
models_state = JLD2.load(gradient_model_state, "state")
Flux.loadmodel!(nns, models_state);
pred_model = ADL.PredictiveModel(
    nns, 
    input_output_map, 
    lags*pd.n_demand+1, 
    pd.n_demand+2*pd.n_zones
)
ADL.set_forecast_model(
    model,
    deepcopy(pred_model)
)
gd_pred_train = model.forecast(X_train')'
gd_cost_train = ADL.compute_cost(model, X_train, Y_train, false, false)
gd_pred_test = model.forecast(X_test')'
gd_cost_test = ADL.compute_cost(model, X_test, Y_test, false, false)

# get NM model
if N_HIDDEN_LAYERS == 0
    models_state = JLD2.load(neldermead_model_state, "state")
    Flux.loadmodel!(nns, models_state);
    pred_model = ADL.PredictiveModel(
        nns, 
        input_output_map, 
        lags*pd.n_demand+1, 
        pd.n_demand+2*pd.n_zones
    )
    ADL.set_forecast_model(
        model,
        deepcopy(pred_model)
    )
    nm_pred_train = model.forecast(X_train')'
    nm_cost_train = ADL.compute_cost(model, X_train, Y_train, false, false)
    nm_pred_test = model.forecast(X_test')'
    nm_cost_test = ADL.compute_cost(model, X_test, Y_test, false, false)
end

dataframe = DataFrame(
    model=String[],
    train_cost=Float64[],
    test_cost=Float64[],
    train_mse=Float64[],
    test_mse=Float64[],
)
push!(dataframe, (
    "LS",
    mean(ls_cost_train),
    mean(ls_cost_test),
    mean(sum((ls_pred_train' .- Y_train') .^2, dims=1)),
    mean(sum((ls_pred_test' .- Y_test') .^2, dims=1)),
))
push!(dataframe, (
    "GD",
    mean(gd_cost_train),
    mean(gd_cost_test),
    mean(sum((gd_pred_train' .- Y_train') .^2, dims=1)),
    mean(sum((gd_pred_test' .- Y_test') .^2, dims=1)),
))
if N_HIDDEN_LAYERS == 0
    push!(dataframe, (
        "NM",
        mean(nm_cost_train),
        mean(nm_cost_test),
        mean(sum((nm_pred_train' .- Y_train') .^2, dims=1)),
        mean(sum((nm_pred_test' .- Y_test') .^2, dims=1)),
    ))
end
println(dataframe)
CSV.write(joinpath(result_path, "costs.csv"), dataframe)

# plot series predictions
N = pd.n_demand
Y = vcat(Y_train, Y_test)
ls_pred = vcat(ls_pred_train, ls_pred_test)
gd_pred = vcat(gd_pred_train, gd_pred_test)
fig = plot(Y[:, 1:N], layout=N, alpha=.7, xticks=false, label="Demand")
plot!(ls_pred[:, 1:N], layout=N, alpha=.7, xticks=false, label="LS")
plot!(gd_pred[:, 1:N], layout=N, alpha=.7, xticks=false, label="GD")
if N_HIDDEN_LAYERS == 0
    nm_pred = vcat(nm_pred_train, nm_pred_test)
    plot!(nm_pred[:, 1:N], layout=N, alpha=.7, xticks=false,  label="NM")
end
plot!(
    ones((2, N)) .* TRAIN_SIZE, 
    vcat(minimum(Y, dims=1), maximum(Y, dims=1)),
    layout=N,
    color=:red,
    label=""
)
plot!(legend=:bottomleft, size=(1200, 800))
plot!(titlefontsize=12, tickfontsize=10, guidefontsize=10, legendfontsize=10)
savefig(fig, joinpath(imgs_path, "predictions.png"))

# plot error histogram
fig = histogram(mean((ls_pred_test - Y_test)[:, 1:pd.n_demand], dims=2), label="LS", alpha=.7)
histogram!(mean((gd_pred_test - Y_test)[:, 1:pd.n_demand], dims=2), label="GD", alpha=.7)
if N_HIDDEN_LAYERS == 0
    histogram!(mean((nm_pred_test - Y_test)[:, 1:pd.n_demand], dims=2), label="NM", alpha=.7)
end
plot!(xlabel="Error", ylabel="Frequency", title="Test Error Histogram")
plot!(legend=:topright, size=(1200, 800))
plot!(titlefontsize=12, tickfontsize=10, guidefontsize=10, legendfontsize=10)
savefig(fig, joinpath(imgs_path, "error_histogram.png"))

# plot costs
fig2 = plot(vcat(ls_cost_train, ls_cost_test), label="LS", alpha=.7)
plot!(vcat(gd_cost_train, gd_cost_test), label="GD", alpha=.7)
if N_HIDDEN_LAYERS == 0
    plot!(vcat(nm_cost_train, nm_cost_test), label="NM", alpha=.7)
end
plot!([TRAIN_SIZE, TRAIN_SIZE], [0, maximum(ls_cost_test)], color=:red, label="")
savefig(fig2, joinpath(imgs_path, "costs.png"))

# detail ls costs
models_state = JLD2.load(pretrained_model_state, "state")
Flux.loadmodel!(nns, models_state);
pred_model = ADL.PredictiveModel(
    nns, 
    input_output_map, 
    lags*pd.n_demand+1, 
    pd.n_demand+2*pd.n_zones
)
ADL.set_forecast_model(
    model,
    deepcopy(pred_model)
)
dataframe = DataFrame(
    data=String[],
    t=Int64[],
    cost_gen=Float64[],
    cost_def=Float64[],
    cost_spl=Float64[],
    cost_total=Float64[],
)
for t=1:size(X_train, 1)
    cost_total = ADL.compute_cost(model, X_train[[t], :], Y_train[[t], :])
    push!(dataframe, (
        "train",
        t,
        value(cost_gen),
        value(cost_def),
        value(cost_spl),
        cost_total
    ))
end
for t=1:size(X_test, 1)
    cost_total = ADL.compute_cost(model, X_test[[t], :], Y_test[[t], :])
    push!(dataframe, (
        "test",
        t,
        value(cost_gen),
        value(cost_def),
        value(cost_spl),
        cost_total
    ))
end
fig = plot(dataframe[:, :cost_gen], label="Gen", alpha=.7)
plot!(dataframe[:, :cost_def], label="Def", alpha=.7)
plot!(dataframe[:, :cost_spl], label="Spl", alpha=.7)
plot!(dataframe[:, :cost_total], label="Total", alpha=.7)
savefig(fig, joinpath(imgs_path, "costs_detail_ls.png"))