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

CASE_NAME = "pglib_opf_case24_ieee_rts"
N_LAGS = 24
N_DEMANDS = 20
N_ZONES = 10
COEF_VARIATION = 0.4
DEFF_COEF = 8.0
SPILL_COEF = 3.0
TEST_SIZE = 7 * 24
SIM_SLICES = 3 * 64

N_HIDDEN_LAYERS = 0
pretrain = false

# .m file path
case_path = joinpath(@__DIR__, "data", CASE_NAME * ".m")

# demand file path
demand_path = joinpath(@__DIR__, "data", "demand.csv")

# results path
result_path = joinpath(@__DIR__, "data", "results", CASE_NAME, "size_$N_HIDDEN_LAYERS")
imgs_path = joinpath(result_path, "imgs")
pretrained_model_state = joinpath(result_path, "pretrain_state.jld2")
gradient_model_state = joinpath(result_path, "model_state_gd.jld2")
neldermead_model_state = joinpath(result_path, "model_state_nm.jld2")

###########################################################################

include("utils/struct.jl")
include("utils/model.jl")
include("utils/data.jl")
include("utils/pretrain.jl")

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
ls_cost_train = ADL.compute_cost(model, X_train, Y_train)
ls_pred_test = model.forecast(X_test')'
ls_cost_test = ADL.compute_cost(model, X_test, Y_test)

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
gd_cost_train = ADL.compute_cost(model, X_train, Y_train)
gd_pred_test = model.forecast(X_test')'
gd_cost_test = ADL.compute_cost(model, X_test, Y_test)

# get NM model
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
nm_cost_train = ADL.compute_cost(model, X_train, Y_train)
nm_pred_test = model.forecast(X_test')'
nm_cost_test = ADL.compute_cost(model, X_test, Y_test)

dataframe = DataFrame(
    model=String[],
    train_cost=Float64[],
    test_cost=Float64[],
    train_mse=Float64[],
    test_mse=Float64[],
)
push!(dataframe, (
    "LS",
    ls_cost_train,
    ls_cost_test,
    mean(sum((ls_pred_train' .- Y_train') .^2, dims=1)),
    mean(sum((ls_pred_test' .- Y_test') .^2, dims=1)),
))
push!(dataframe, (
    "GD",
    gd_cost_train,
    gd_cost_test,
    mean(sum((gd_pred_train' .- Y_train') .^2, dims=1)),
    mean(sum((gd_pred_test' .- Y_test') .^2, dims=1)),
))
push!(dataframe, (
    "NM",
    nm_cost_train,
    nm_cost_test,
    mean(sum((nm_pred_train' .- Y_train') .^2, dims=1)),
    mean(sum((nm_pred_test' .- Y_test') .^2, dims=1)),
))
println(dataframe)
CSV.write(joinpath(result_path, "costs.csv"), dataframe)

N = pd.n_demand
Y = vcat(Y_train, Y_test)
ls_pred = vcat(ls_pred_train, ls_pred_test)
gd_pred = vcat(gd_pred_train, gd_pred_test)
nm_pred = vcat(nm_pred_train, nm_pred_test)
fig = plot(Y[:, 1:N], layout=N, alpha=.7, xticks=false, label="Demand")
plot!(ls_pred[:, 1:N], layout=N, alpha=.7, xticks=false, label="LS")
plot!(gd_pred[:, 1:N], layout=N, alpha=.7, xticks=false, label="GD")
plot!(nm_pred[:, 1:N], layout=N, alpha=.7, xticks=false,  label="NM")
plot!(legend=:topleft, size=(1200, 800))
plot!(titlefontsize=12, tickfontsize=10, guidefontsize=10, legendfontsize=10)
savefig(fig, joinpath(imgs_path, "predictions.png"))
