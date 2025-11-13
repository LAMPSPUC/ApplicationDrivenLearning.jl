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

if gradient_mode
    # gradient mode
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
            time_limit=TIME_LIMIT
        )
    )
    println("GradientMode training time: $(time() - time1)")

    gd_pred = model.forecast(X_train')'
    gd_mse = sum((gd_pred' .- Y_train') .^2, dims=1) |> mean
    gd_cost = ADL.compute_cost(model, X_train, Y_train)

    println("OPT-GD MSE: $gd_mse")
    println("OPT-GD: $gd_cost")

elseif neldermead_mode
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
    nm_mse = sum((nm_pred' .- Y_train') .^2, dims=1) |> mean  # 1247.29
    nm_cost = ADL.compute_cost(model, X_train, Y_train)  # 6288.93

    println("OPT-NM MSE: $nm_mse")
    println("OPT-NM: $nm_cost")
else
    println("Optimization mode not defined. Skipping optimization.")
end

