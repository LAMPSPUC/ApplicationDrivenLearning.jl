# JQM.mpiexec(exe -> run(`$exe -n 12 $(Base.julia_cmd()) --project main.jl`))

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
import JobQueueMPI
using ApplicationDrivenLearning

JQM = JobQueueMPI

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
            ApplicationDrivenLearning.GradientMPIMode;
            rule=Flux.Adam(LEARNING_RATE), 
            epochs=N_EPOCHS,
            compute_cost_every=COMPUTE_EVERY,
            batch_size=BATCH_SIZE,
            time_limit=TIME_LIMIT,
            mpi_finalize=false
        )
    )
    if JQM.is_controller_process()
        print("GradientMPIMode training time: $(time() - time1)")

        gd_pred = model.forecast(X_train')'
        gd_mse = sum((gd_pred' .- Y_train') .^2, dims=1) |> mean
        gd_cost = ADL.compute_cost(model, X_train, Y_train)

        println("OPT-GD MSE: $gd_mse")
        println("OPT-GD: $gd_cost")
    end

elseif neldermead_mode
    # nelder mead mode
    time1 = time()
    sol = ApplicationDrivenLearning.train!(
        model,
        X_train,
        Y_train,
        ApplicationDrivenLearning.Options(
            ApplicationDrivenLearning.NelderMeadMPIMode;
            iterations=N_EPOCHS, 
            show_trace=true, 
            show_every=30,
            time_limit=TIME_LIMIT,
            mpi_finalize=false
        )
    )
    
    if JQM.is_controller_process()
        println("NelderMeadMPIMode training time: $(time() - time1)")

        nm_pred = model.forecast(X_train')'
        nm_mse = sum((nm_pred' .- Y_train') .^2, dims=1) |> mean  # 1247.29
        nm_cost = ADL.compute_cost(model, X_train, Y_train)  # 6288.93

        println("OPT-NM MSE: $nm_mse")
        println("OPT-NM: $nm_cost")
    end
else
    println("Optimization mode not defined. Skipping optimization.")
end

if (gradient_mode | neldermead_mode) 
    if JQM.is_controller_process()
        # store models states
        JLD2.jldsave(
            final_model_state; 
            state=Flux.state(model.forecast.networks)
        )
    end
    
    # MPI finalize
    JQM.mpi_finalize()

end

