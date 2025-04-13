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

################################ VARIABLES ################################

pretrain = false
gradient_mode = false
neldermead_mode = true
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
HIDDEN_SIZE = 16

# pretrain parameters
PRETRAIN_EPOCHS = 30_000
PRETRAIN_MAX_TIME = 60
PRETRAIN_LEARNING_RATE = 1e-1
PRETRAIN_BATCH_SIZE = -1

# opt train parameters
N_EPOCHS = 30_000
BATCH_SIZE = -1
LEARNING_RATE = 1e-3
COMPUTE_EVERY = 1
TIME_LIMIT = 60*9

# .m file path
case_path = joinpath(@__DIR__, "data", CASE_NAME * ".m")

# demand file path
demand_path = joinpath(@__DIR__, "data", "demand.csv")

# results path
result_path = joinpath(@__DIR__, "data", "results", CASE_NAME, "size_$N_HIDDEN_LAYERS")
imgs_path = joinpath(result_path, "imgs")
pretrained_model_state = joinpath(result_path, "pretrain_state.jld2")
if gradient_mode
    final_model_state = joinpath(result_path, "model_state_gd.jld2")
elseif neldermead_mode
    final_model_state = joinpath(result_path, "model_state_nm.jld2")
else 
    final_model_state = nothing
end

# create folders if necessary
if !(isdir(result_path))
    mkpath(result_path)
    mkpath(imgs_path)
end

###########################################################################

include("utils/struct.jl")
include("utils/model.jl")
include("utils/data.jl")
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

