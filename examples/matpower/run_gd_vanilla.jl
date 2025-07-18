# This script runs the Nelder-Mead optimization algorithm (without MPI) on a predictive model

import Parameters
using JuMP
import DelimitedFiles
import Random
using Statistics
import OffsetArrays: OffsetArray
import Gurobi
using Flux
import NNlib
import Distributions
import JLD2
using ApplicationDrivenLearning

include("config.jl")
include("utils/struct.jl")
include("utils/data.jl")
include("utils/model.jl")
include("utils/pretrain.jl")

# least-squares model
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

ls_pred = model.forecast(X_train')'
ls_mse = sum((ls_pred' .- Y_train') .^2, dims=1) |> mean
println("LS MSE: $ls_mse")

# nelder mead mode
time1 = time()
sol = ApplicationDrivenLearning.train!(
    model,
    X_train,
    Y_train,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.GradientMode;
        rule=Flux.Adam(LEARNING_RATE), 
        epochs=N_EPOCHS,
        compute_cost_every=COMPUTE_EVERY,
        batch_size=BATCH_SIZE,
        time_limit=TIME_LIMIT,
    )
)

println("GradientMode training time: $(time() - time1)")

gd_pred = model.forecast(X_train')'
gd_mse = sum((gd_pred' .- Y_train') .^2, dims=1) |> mean
gd_cost = ADL.compute_cost(model, X_train, Y_train)

println("OPT-GD MSE (train): $gd_mse")
println("OPT-GD (train): $gd_cost")

gd_pred = model.forecast(X_test')'
gd_mse = sum((gd_pred' .- Y_test') .^2, dims=1) |> mean
gd_cost = ADL.compute_cost(model, X_test, Y_test)

println("OPT-GD MSE (test): $gd_mse")
println("OPT-GD (test): $gd_cost")
