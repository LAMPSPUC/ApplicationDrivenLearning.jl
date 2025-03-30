import CSV
import Flux
import HiGHS
import Random
using JuMP
using DataFrames

include("..//..//..//ApplicationDrivenLearning.jl_DEV//src//ApplicationDrivenLearning.jl")
using .ApplicationDrivenLearning


Random.seed!(0)

# paths
DATA_PATH = joinpath(@__DIR__, "data")
INPT_PATH = joinpath(DATA_PATH, "input")

# load data
c_df = CSV.read(joinpath(INPT_PATH, "c.csv"), DataFrame)
x_df = CSV.read(joinpath(INPT_PATH, "x.csv"), DataFrame)

# split train and test
x_train = Matrix(x_df)[1:100, :] .|> Float32
x_test  = Matrix(x_df)[101:end, :] .|> Float32
c_train = Matrix(c_df)[1:100, :] .|> Float64
c_test  = Matrix(c_df)[101:end, :] .|> Float64

# parameters
T, p = size(x_df)  # (300, 5)
m = size(c_df, 2)  # 40
grid = (5, 5)
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
@constraint(Plan(optmodel), x_plan_set .>= 0)
@constraint(Assess(optmodel), x_assess_set .>= 0)
@constraint(Plan(optmodel), x_plan_set .<= 1)
@constraint(Assess(optmodel), x_assess_set .<= 1)
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
            @constraint(Plan(optmodel), expr_plan == -1)
            @constraint(Assess(optmodel), expr_assess == -1)
        # sink
        elseif (i == grid[1]) && (j == grid[2])
            @constraint(Plan(optmodel), expr_plan == 1)
            @constraint(Assess(optmodel), expr_assess == 1)
        # transition
        else
            @constraint(Plan(optmodel), expr_plan == 0)
            @constraint(Assess(optmodel), expr_assess == 0)
        end
    end
end
@objective(Plan(optmodel), Min, c_plan_set'x_plan_set)
@objective(Assess(optmodel), Min, c_assess_set'x_assess_set)
set_optimizer(optmodel, HiGHS.Optimizer)
set_silent(optmodel)

# linear model
reg = Flux.Chain(Flux.Dense(p => m))
train_data = Flux.DataLoader((x_train', c_train'), batchsize=32)
opt_state = Flux.setup(Flux.Adam(1e-2), reg)
for epoch=1:100
    Flux.train!(reg, train_data, opt_state) do m, x, y
        sum((m(x) .- y).^2)
    end
    if epoch % 10 == 0
        println("Epoch $epoch | Squared-error: $(sum((reg(x_train') .- c_train').^2))")
    end
end
ApplicationDrivenLearning.set_forecast_model(optmodel, reg)
println("LS model train cost: $(compute_cost(optmodel, x_train, c_train, false))")

t0 = time()
nm_sol = ApplicationDrivenLearning.train!(
    optmodel, x_train[ind, :], c_train[ind, :],
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.NelderMeadMode,
        iterations=30, 
        show_trace=true, 
        show_every=1,
        time_limit=60,
    )
)
println("Elapsed time: $(round(time() - t0, digits=2)) seconds")

costs_to_compare = Matrix(
    CSV.read(joinpath(DATA_PATH, "costs_to_compare.csv"), DataFrame)
)[:, 1]
solutions_to_compare = Matrix(
    CSV.read(joinpath(DATA_PATH, "solutions_to_compare.csv"), DataFrame)
)

test_costs = zeros(size(c_test, 1))
test_solutions = zeros(size(c_test))
for i=1:size(x_test, 1)
    c = compute_cost(optmodel, x_test[[i], :], c_test[[i], :])
    sol = value.(ApplicationDrivenLearning.assess_policy_vars(optmodel))
    test_costs[i] = c
    test_solutions[i, :] .= sol
end

println("Avg test cost (base) = $(sum(costs_to_compare) / size(c_test, 1))")
println("Avg test cost (model) = $(sum(test_costs) / size(c_test, 1))")
