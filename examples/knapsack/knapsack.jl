import CSV
import Flux
import Random
using Gurobi
using JuMP
using DataFrames
using ApplicationDrivenLearning


Random.seed!(0)

# paths
DATA_PATH = joinpath(@__DIR__, "data")
INPT_PATH = joinpath(DATA_PATH, "input")

# load data
weights_df = CSV.read(joinpath(INPT_PATH, "weights.csv"), DataFrame)
c_df = CSV.read(joinpath(INPT_PATH, "c.csv"), DataFrame)
x_df = CSV.read(joinpath(INPT_PATH, "x.csv"), DataFrame)

# split train and test
x_train = Matrix(x_df)[1:100, :] .|> Float32
x_test  = Matrix(x_df)[101:end, :] .|> Float32
c_train = Matrix(c_df)[1:100, :] .|> Float32
c_test  = Matrix(c_df)[101:end, :] .|> Float32

# parameters
T, p = size(x_df)  # (300, 5)
I, d = size(weights_df)  # (32, 2)
W = Matrix(weights_df)
caps = ones(d) * 20 # capacity

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

# linear model
reg = Flux.Dense(p => I)
train_data = Flux.DataLoader((x_train', c_train'), batchsize=32)
opt_state = Flux.setup(Flux.Adam(1e-1), reg)
for epoch=1:30
    Flux.train!(reg, train_data, opt_state) do m, x, y
        sum((m(x) .- y).^2)
    end
    if epoch % 10 == 0
        println("Epoch $epoch | Squared-error: $(sum((reg(x_train') .- c_train').^2))")
    end
end
ApplicationDrivenLearning.set_forecast_model(optmodel, reg)
println("LS model train cost: $(ApplicationDrivenLearning.compute_cost(optmodel, x_train, c_train, false))")

# train with nelder mead
nm_sol = ApplicationDrivenLearning.train!(
    optmodel, x_train, c_train,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.NelderMeadMode,
        iterations=100, 
        show_trace=true, 
        show_every=1,
        time_limit=300,
    )
)


costs_to_compare = Matrix(
    CSV.read(joinpath(DATA_PATH, "costs_to_compare.csv"), DataFrame)
)[:, 1]
solutions_to_compare = Matrix(
    CSV.read(joinpath(DATA_PATH, "solutions_to_compare.csv"), DataFrame)
)

test_costs = zeros(size(c_test, 1))
test_solutions = zeros(size(c_test))
for i=1:size(x_test, 1)
    c = -ApplicationDrivenLearning.compute_cost(optmodel, x_test[[i], :], c_test[[i], :])
    sol = value.(ApplicationDrivenLearning.assess_policy_vars(optmodel))
    test_costs[i] = c
    test_solutions[i, :] .= sol
end

println("Avg test cost (base) = $(sum(costs_to_compare) / size(c_test, 1))")  # -36.3
println("Avg test cost (model) = $(sum(test_costs) / size(c_test, 1))")  # -36.745
