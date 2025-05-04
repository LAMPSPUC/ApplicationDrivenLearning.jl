import CSV
import Flux
import Gurobi
import Random
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
grid = (5, 5)

function get_data()
    # load data
    c_df = CSV.read(joinpath(INPT_PATH, "c.csv"), DataFrame)
    x_df = CSV.read(joinpath(INPT_PATH, "x.csv"), DataFrame)

    # split train and test
    x_train = Matrix(x_df)[1:train_size, :] .|> Float32
    x_test  = Matrix(x_df)[train_size+1:end, :] .|> Float32
    c_train = Matrix(c_df)[1:train_size, :] .|> Float32
    c_test  = Matrix(c_df)[train_size+1:end, :] .|> Float32

    return (x_train, x_test, c_train, c_test)
end

function pretrain_model(X, C, epochs=100, lr=1e-2, batchsize=32, verbose=true)
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

function get_optmodel(c_train)
    m = size(c_train, 2)

    # construct arcs (edges)
    arcs = Matrix{Float64}(undef, m, 2)
    im = 1
    for i=1:grid[1]
        # edges on rows
        for j=1:grid[2]-1
            v = (i-1) * grid[2] + j
            arcs[im, :] = [v, v+1]
            im += 1
        end
        # edges in columns
        if i != grid[1]
            for j=1:grid[2]
                v = (i-1) * grid[2] + j
                arcs[im, :] = [v, v+grid[2]]
                im += 1
            end
        end  
    end

    # init optimization model
    optmodel = ApplicationDrivenLearning.Model()
    @variables(optmodel, begin
        c[1:m], ApplicationDrivenLearning.Forecast
        x[1:m], ApplicationDrivenLearning.Policy
    end)
    x_plan_set = [ix.plan for ix in x]
    c_plan_set = [ic.plan for ic in c]
    x_assess_set = [ix.assess for ix in x]
    c_assess_set = [ic.assess for ic in c]
    @constraint(ApplicationDrivenLearning.Plan(optmodel), x_plan_set .>= 0)
    @constraint(ApplicationDrivenLearning.Assess(optmodel), x_assess_set .>= 0)
    @constraint(ApplicationDrivenLearning.Plan(optmodel), x_plan_set .<= 1)
    @constraint(ApplicationDrivenLearning.Assess(optmodel), x_assess_set .<= 1)
    for i=1:grid[1]
        for j=1:grid[2]
            v = (i-1)*grid[2] + j
            expr_plan = 0
            expr_assess = 0
            for im=1:m
                e = arcs[im, :]
                # flow in
                if v == e[2]
                    expr_plan += x_plan_set[im]
                    expr_assess += x_assess_set[im]
                end
                # flow out
                if v == e[1]
                    expr_plan -= x_plan_set[im]
                    expr_assess -= x_assess_set[im]
                end
            end
            # source
            if (i == 1) && (j == 1)
                @constraint(ApplicationDrivenLearning.Plan(optmodel), expr_plan == -1)
                @constraint(ApplicationDrivenLearning.Assess(optmodel), expr_assess == -1)
            # sink
            elseif (i == grid[1]) && (j == grid[2])
                @constraint(ApplicationDrivenLearning.Plan(optmodel), expr_plan == 1)
                @constraint(ApplicationDrivenLearning.Assess(optmodel), expr_assess == 1)
            # transition
            else
                @constraint(ApplicationDrivenLearning.Plan(optmodel), expr_plan == 0)
                @constraint(ApplicationDrivenLearning.Assess(optmodel), expr_assess == 0)
            end
        end
    end
    @objective(ApplicationDrivenLearning.Plan(optmodel), Min, c_plan_set'x_plan_set)
    @objective(ApplicationDrivenLearning.ApplicationDrivenLearning.Assess(optmodel), Min, c_assess_set'x_assess_set)
    set_optimizer(optmodel, Gurobi.Optimizer)
    set_silent(optmodel)

    return optmodel
end

# load data
(x_train, x_test, c_train, c_test) = get_data()

# pretrain model
reg = pretrain_model(x_train, c_train)

# set forecast model
optmodel = get_optmodel(c_train);
ApplicationDrivenLearning.set_forecast_model(optmodel, reg);

# train with nelder mead
nm_sol = ApplicationDrivenLearning.train!(
    optmodel, x_train, c_train,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.NelderMeadMode,
        iterations=1_000, 
        show_trace=true, 
        show_every=1,
        time_limit=60,
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
test_cost_df = DataFrame(:test_cost => costs, :opt_cost => opt_costs)

# save results
CSV.write(joinpath(OUTP_PATH, "costs.csv"), test_cost_df)
CSV.write(joinpath(OUTP_PATH, "predictions.csv"), DataFrame(preds, :auto))
CSV.write(joinpath(OUTP_PATH, "solutions.csv"), DataFrame(solutions, :auto))