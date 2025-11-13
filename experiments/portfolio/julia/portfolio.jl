import CSV
import Flux
import Optim
import Gurobi
import Random
using Statistics
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

function get_data()
    # load data
    cov_df = CSV.read(joinpath(INPT_PATH, "cov.csv"), DataFrame)
    c_df = CSV.read(joinpath(INPT_PATH, "c.csv"), DataFrame)
    x_df = CSV.read(joinpath(INPT_PATH, "x.csv"), DataFrame)

    cov = Matrix(cov_df) .|> Float32

    # split train and test
    x_train = Matrix(x_df)[1:train_size, :] .|> Float32
    x_test  = Matrix(x_df)[train_size+1:end, :] .|> Float32
    c_train = Matrix(c_df)[1:train_size, :] .|> Float32
    c_test  = Matrix(c_df)[train_size+1:end, :] .|> Float32

    return (cov, x_train, x_test, c_train, c_test)
end

function pretrain_model(X, C, epochs=1000, lr=1e-3, batchsize=32, verbose=true)
    # linear model
    reg = Flux.Dense(size(X, 2) => size(C, 2))
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
        c = ApplicationDrivenLearning.compute_cost(optmodel, X[[i], :], C[[i], :])
        pred = optmodel.forecast(X[[i], :]')
        sol = value.(ApplicationDrivenLearning.assess_policy_vars(optmodel))
        costs[i] = c
        preds[i, :] .= pred
        solutions[i, :] .= sol
    end
    return costs, preds, solutions
end

function get_optmodel(cov, gamma=2.25)
    m = size(cov, 1)
    risk_level = gamma * mean(cov)

    # init optimization model
    optmodel = ApplicationDrivenLearning.Model()
    @variables(optmodel, begin
        c[1:m], ApplicationDrivenLearning.Forecast
        x[1:m], ApplicationDrivenLearning.Policy
    end)

    # plan
    x_plan_set = [ix.plan for ix in x]
    c_plan_set = [ic.plan for ic in c]
    @constraint(ApplicationDrivenLearning.Plan(optmodel), x_plan_set .>= 0)
    @constraint(ApplicationDrivenLearning.Plan(optmodel), x_plan_set .<= 1)
    @constraint(ApplicationDrivenLearning.Plan(optmodel), sum(x_plan_set) == 1)
    @constraint(ApplicationDrivenLearning.Plan(optmodel), x_plan_set'cov*x_plan_set <= risk_level)
    @objective(ApplicationDrivenLearning.Plan(optmodel), Min, -c_plan_set'x_plan_set)

    # assess
    x_assess_set = [ix.assess for ix in x]
    c_assess_set = [ic.assess for ic in c]
    @constraint(ApplicationDrivenLearning.Assess(optmodel), x_assess_set .>= 0)
    @constraint(ApplicationDrivenLearning.Assess(optmodel), x_assess_set .<= 1)
    @constraint(ApplicationDrivenLearning.Assess(optmodel), sum(x_assess_set) == 1)
    @constraint(ApplicationDrivenLearning.Assess(optmodel), x_assess_set'cov*x_assess_set <= risk_level)
    @objective(ApplicationDrivenLearning.Assess(optmodel), Min, -c_assess_set'x_assess_set)

    set_optimizer(optmodel, Gurobi.Optimizer)
    set_silent(optmodel)

    return optmodel
end

# load data
(cov, x_train, x_test, c_train, c_test) = get_data()

# pretrain model
reg = pretrain_model(x_train, c_train)

# set forecast model
optmodel = get_optmodel(cov);
ApplicationDrivenLearning.set_forecast_model(optmodel, reg);

ls_cost_train = ApplicationDrivenLearning.compute_cost(optmodel, x_train, c_train, false, false)
ls_cost_test = ApplicationDrivenLearning.compute_cost(optmodel, x_test, c_test, false, false)

# train with nelder mead
nm_sol = ApplicationDrivenLearning.train!(
    optmodel, x_train, c_train,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.NelderMeadMode,
        initial_simplex=Optim.AffineSimplexer(0.9, 0.1),
        iterations=100,
        show_trace=true, 
        show_every=1,
        time_limit=10*60,
    )
)

# get solution
costs, preds, solutions = get_solution(optmodel, x_test, c_test)

# get optimal costs
opt_costs = zeros(size(c_test, 1))
for i=1:size(x_test, 1)
    y = c_test[i, :]
    opt_costs[i] = ApplicationDrivenLearning.compute_single_step_cost(optmodel, y, y)
end
test_cost_df = DataFrame(
    :test_cost => costs, 
    :opt_cost => opt_costs,
    :ls_cost => ls_cost_test
)

# save results
CSV.write(joinpath(OUTP_PATH, "costs.csv"), test_cost_df)
CSV.write(joinpath(OUTP_PATH, "predictions.csv"), DataFrame(preds, :auto))
CSV.write(joinpath(OUTP_PATH, "solutions.csv"), DataFrame(solutions, :auto))

# test pyepo results
RES2_PATH = joinpath(DATA_PATH, "pyepo_result")
preds_pyepo = Matrix(CSV.read(joinpath(RES2_PATH, "predictions.csv"), DataFrame))
costs_pyepo = Matrix(CSV.read(joinpath(RES2_PATH, "costs.csv"), DataFrame))
solutions_pyepo = Matrix(CSV.read(joinpath(RES2_PATH, "solutions.csv"), DataFrame))

pyepo_costs = zeros(size(c_test, 1))
for i=1:size(c_test, 1)
    pyepo_costs[i] = ApplicationDrivenLearning.compute_single_step_cost(optmodel, c_test[i, :], preds_pyepo[i, :])
end
