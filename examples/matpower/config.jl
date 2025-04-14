# this config file is used to set the parameters for the experiment

run_mode = 2  # 1: pretrain, 2: gradient, 3: neldermead

pretrain = false
gradient_mode = false
neldermead_mode = false

if run_mode == 1
    pretrain = true
    println("Pretrain mode")
elseif run_mode == 2
    gradient_mode = true
    println("Gradient mode")
elseif run_mode == 3
    neldermead_mode = true
    println("NelderMead mode")
else
    error("Invalid option")
end

CASE_NAME = "pglib_opf_case24_ieee_rts"
N_LAGS = 24
N_DEMANDS = 20
N_ZONES = 10
COEF_VARIATION = 0.4
DEFF_COEF = 8.0
SPILL_COEF = 3.0
TRAIN_SIZE = 29 * 24
TEST_SIZE = 7 * 24
SIM_SLICES = 3 * 64

N_HIDDEN_LAYERS = 0
HIDDEN_SIZE = 64

# pretrain parameters
PRETRAIN_EPOCHS = 10_000
PRETRAIN_MAX_TIME = 60
PRETRAIN_LEARNING_RATE = 1e-2
PRETRAIN_BATCH_SIZE = -1

# opt train parameters
N_EPOCHS = 10_000
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
COMPUTE_EVERY = 10
TIME_LIMIT = 60*9

# .m file path
case_path = joinpath(@__DIR__, "data", "cases", CASE_NAME * ".m")

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