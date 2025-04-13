import Gurobi
import ApplicationDrivenLearning
using JuMP

model = ApplicationDrivenLearning.Model()
@variable(model, 0 <= z <= 4, ApplicationDrivenLearning.Policy)
@variable(model, y, ApplicationDrivenLearning.Forecast)

@variables(ApplicationDrivenLearning.Plan(model), begin
    cp1 >= 0
    cp2 >= 0
end)
@constraints(ApplicationDrivenLearning.Plan(model), begin
    cp1 >= 100 * (y.plan - z.plan)
    cp2 >= 20 * (z.plan - y.plan)
end)
@objective(ApplicationDrivenLearning.Plan(model), Min, 10*z.plan + cp1 + cp2)

@variables(ApplicationDrivenLearning.Assess(model), begin
    ca1 >= 0
    ca2 >= 0
end)
@constraints(ApplicationDrivenLearning.Assess(model), begin
    ca1 >= 100 * (y.assess - z.assess)
    ca2 >= 20 * (z.assess - y.assess)
end)
@objective(ApplicationDrivenLearning.Assess(model), Min, 10*z.assess + ca1 + ca2)

using Flux
predictive = Dense(1 => 1; bias=false)
ApplicationDrivenLearning.set_forecast_model(model, predictive)

X = ones(30, 1)
Y = rand([0, 2], (30, 1))
set_optimizer(model, Gurobi.Optimizer)
mode = ApplicationDrivenLearning.Options(
    ApplicationDrivenLearning.NelderMeadMode
)
solution = ApplicationDrivenLearning.train!(
    model, X, Y, mode
)

println("Previsão final = $(solution.params)")
println("Custo final estimado = $(solution.cost)")
