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
using BilevelJuMP
using Plots
import Distributions
import JLD2
import JobQueueMPI
using ApplicationDrivenLearning

JQM = JobQueueMPI

################################ VARIABLES ################################

pretrain = false
gradient_mode = true
neldermead_mode = false
CASE_NAME = "pglib_opf_case300_ieee"
N_LAGS = 24
N_DEMANDS = 20
N_ZONES = 10
COEF_VARIATION = 0.4
DEFF_COEF = 8.0
SPILL_COEF = 3.0
TEST_SIZE = 7 * 24
SIM_SLICES = 3 * 64
SOLVE_TIME_LIMIT = 30

# pretrain parameters
PRETRAIN_EPOCHS = 30_000
PRETRAIN_MAX_TIME = 60
PRETRAIN_LEARNING_RATE = 1e-1
PRETRAIN_BATCH_SIZE = -1

# opt train parameters
N_EPOCHS = 300
BATCH_SIZE = -1
LEARNING_RATE = 1e-3
COMPUTE_EVERY = 1
TIME_LIMIT = 600

# .m file path
case_path = joinpath(@__DIR__, "data", CASE_NAME * ".m")

# demand csv file path
demand_path = joinpath(@__DIR__, "data", "demand.csv")

result_path = joinpath(@__DIR__, "data", "results", CASE_NAME)
if !(isdir(result_path))
    mkpath(result_path)
end
imgs_path = joinpath(result_path, "imgs")
if !(isdir(imgs_path))
    mkpath(imgs_path)
end
pretrained_model_state = joinpath(result_path, "pretrain_state.jld2")
if gradient_mode
    final_model_state = joinpath(result_path, "model_state_gd.jld2")
else
    final_model_state = joinpath(result_path, "model_state_nm.jld2")
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

ls_pred = model.forecast(X')'
ls_mse = sum((ls_pred' .- Y') .^2, dims=1) |> mean
# ls_cost = ADL.compute_cost(model, X, Y)

println("LS MSE: $ls_mse")
# println("LS Cost: $ls_cost")

if gradient_mode
    # gradient mode
    time1 = time()
    sol = ApplicationDrivenLearning.train!(
        model,
        X,
        Y,
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

        gd_pred = model.forecast(X')'
        gd_mse = sum((gd_pred' .- Y') .^2, dims=1) |> mean
        gd_cost = ADL.compute_cost(model, X, Y)

        println("OPT-GD MSE: $gd_mse")
        println("OPT-GD: $gd_cost")
    end

elseif neldermead_mode
    # nelder mead mode
    time1 = time()
    sol = ApplicationDrivenLearning.train!(
        model,
        X,
        Y,
        ApplicationDrivenLearning.Options(
            ApplicationDrivenLearning.NelderMeadMPIMode;
            iterations=N_EPOCHS, 
            show_trace=true, 
            show_every=COMPUTE_EVERY,
            time_limit=TIME_LIMIT,
            mpi_finalize=false
        )
    )
    
    if JQM.is_controller_process()
        println("NelderMeadMPIMode training time: $(time() - time1)")

        nm_pred = model.forecast(X')'
        nm_mse = sum((nm_pred' .- Y') .^2, dims=1) |> mean  # 1247.29
        nm_cost = ADL.compute_cost(model, X, Y)  # 6288.93

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

