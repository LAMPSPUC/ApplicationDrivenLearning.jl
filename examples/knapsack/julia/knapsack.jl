import CSV
import Flux
import Random
using Gurobi
using JuMP
using DataFrames
using ApplicationDrivenLearning


Random.seed!(0)

# paths
DATA_PATH = joinpath(@__DIR__, "..//data")
INPT_PATH = joinpath(DATA_PATH, "input")
OUTP_PATH = joinpath(DATA_PATH, "adl_result")

# parameters
train_size = 100
I = 8 # number of items

function get_data()
    # load data
    weights_df = CSV.read(joinpath(INPT_PATH, "weights.csv"), DataFrame)
    c_df = CSV.read(joinpath(INPT_PATH, "c.csv"), DataFrame)
    x_df = CSV.read(joinpath(INPT_PATH, "x.csv"), DataFrame)

    # split train and test
    x_train = Matrix(x_df)[1:train_size, :] .|> Float32
    x_test  = Matrix(x_df)[train_size+1:end, :] .|> Float32
    c_train = Matrix(c_df)[1:train_size, :] .|> Float32
    c_test  = Matrix(c_df)[train_size+1:end, :] .|> Float32
    W = Matrix(weights_df)

    # other data
    caps = ones(size(W, 2)) * 20 # capacity

    return (W, caps, x_train, x_test, c_train, c_test)
end

function pretrain_model(X, C, epochs=100, lr=1e-1, batchsize=32, verbose=true)
    # linear model
    reg = Flux.Dense(size(X, 2) => I)
    train_data = Flux.DataLoader((X', C'), batchsize=batchsize)
    opt_state = Flux.setup(Flux.Adam(lr), reg)
    for epoch=1:epochs
        Flux.train!(reg, train_data, opt_state) do m, x, y
            sum((m(x) .- y).^2)
        end
        if verbose && (epoch % 10 == 0)
            println("Epoch $epoch | Squared-error: $(sum((reg(X') .- C').^2))")
        end
    end
    return reg
end

function get_solution(optmodel, X, C)
    costs = zeros(size(C, 1))
    preds = zeros(size(C))
    solutions = zeros(size(C))
    for i=1:size(X, 1)
        c = -ApplicationDrivenLearning.compute_cost(optmodel, X[[i], :], C[[i], :])
        pred = optmodel.forecast(X[[i], :]')
        sol = value.(ApplicationDrivenLearning.assess_policy_vars(optmodel))
        costs[i] = c
        preds[i, :] .= pred
        solutions[i, :] .= sol
    end
    return costs, preds, solutions
end

function get_optmodel(W, caps)
    # init optimization model
    optmodel = ApplicationDrivenLearning.Model()
    @variables(optmodel, begin
        c[1:I], ApplicationDrivenLearning.Forecast
        x[1:I], ApplicationDrivenLearning.Policy, Bin
    end)
    x_plan_set = [ix.plan for ix in x]
    c_plan_set = [ic.plan for ic in c]
    @constraint(ApplicationDrivenLearning.Plan(optmodel), x_plan_set'W .<= caps')
    @objective(ApplicationDrivenLearning.Plan(optmodel), Min, -c_plan_set'x_plan_set)
    x_assess_set = [ix.assess for ix in x]
    c_assess_set = [ic.assess for ic in c]
    @constraint(ApplicationDrivenLearning.Assess(optmodel), x_assess_set'W .<= caps')
    @objective(ApplicationDrivenLearning.Assess(optmodel), Min, -c_assess_set'x_assess_set)
    set_optimizer(optmodel, Gurobi.Optimizer)
    set_silent(optmodel)
    return optmodel
end

# load data
(W, caps, x_train, x_test, c_train, c_test) = get_data()

# pretrain model
reg = pretrain_model(x_train, c_train)

# set forecast model
optmodel = get_optmodel(W, caps);
ApplicationDrivenLearning.set_forecast_model(optmodel, reg);

# train with nelder mead
nm_sol = ApplicationDrivenLearning.train!(
    optmodel, x_train, c_train,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.NelderMeadMode,
        iterations=1_000,
        show_trace=true,
        show_every=5,
        time_limit=60,
        g_tol=1e-2
    )
)

# get test costs
test_costs, test_predictions, test_solutions = get_solution(optmodel, x_test, c_test)

# get optimzal costs
opt_costs = zeros(size(c_test, 1))
for i=1:size(x_test, 1)
    y = c_test[i, :]
    opt_costs[i] = -ApplicationDrivenLearning.compute_single_step_cost(optmodel, y, y)
end
test_cost_df = DataFrame(:test_cost => test_costs, :opt_cost => opt_costs)

# save results
CSV.write(joinpath(OUTP_PATH, "costs.csv"), test_cost_df)
CSV.write(joinpath(OUTP_PATH, "predictions.csv"), DataFrame(test_predictions, :auto))
CSV.write(joinpath(OUTP_PATH, "solutions.csv"), DataFrame(test_solutions, :auto))
