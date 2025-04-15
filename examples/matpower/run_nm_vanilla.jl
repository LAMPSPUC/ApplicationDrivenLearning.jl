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
        ApplicationDrivenLearning.NelderMeadMode;
        iterations=N_EPOCHS, 
        show_trace=true,
        show_every=30,
        time_limit=TIME_LIMIT
    )
)

println("NelderMeadMode training time: $(time() - time1)")

nm_pred = model.forecast(X_train')'
nm_mse = sum((nm_pred' .- Y_train') .^2, dims=1) |> mean
nm_cost = ADL.compute_cost(model, X_train, Y_train)

println("OPT-NM MSE (train): $nm_mse")
println("OPT-NM (train): $nm_cost")

nm_pred = model.forecast(X_test')'
nm_mse = sum((nm_pred' .- Y_test') .^2, dims=1) |> mean
nm_cost = ADL.compute_cost(model, X_test, Y_test)

println("OPT-NM MSE (test): $nm_mse")
println("OPT-NM (test): $nm_cost")
