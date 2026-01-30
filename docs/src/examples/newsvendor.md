# Newsvendor Problem

This example shows how to use the ApplicationDrivenLearning.jl package to solve a newsvendor problem.

The newsvendor problem is a classic inventory management problem where a company must decide how many units to produce to meet demand. The company wants to minimize the cost of holding excess inventory while also minimizing the cost of lost sales. 

## The model

For this problem, both plan and assess models are defined as:

```math
\min c.x - q.y - r.w \\
\text{s.t.} \quad x,y,w \geq 0 \\
y + w \leq x \\
y \leq d
```

The plan model will be run considering the demand `d` equal to the output of the forecast model, defining the policy `x` that will be used in the assess model. The assess model will be run considering the demand `d` equal to the actual demand.

## Data

We will model two separate items. One of the items presents a low overstocking cost and the other presents a low under-stocking cost, generating different incentives that a decision maker could explore. This is achieved manipulating the costs:

```math
i=1 \longrightarrow c=10; \quad q = 19;\quad r = 9
```

```math
i=2 \longrightarrow c=10; \quad q = 11;\quad r = 1
```

The demand series for both items will be generated using a discrete uniform distribution.

Let's start by loading the necessary packages and defining the problem parameters.

```julia
using Flux
using JuMP
using Random
using Gurobi
using ApplicationDrivenLearning

ADL = ApplicationDrivenLearning

# data
Random.seed!(123)
c = [10, 10]
q = [19, 11]
r = [9, 1]
```

Now, we can initialize the application driven learning model, build the plan and assess models and set the forecast model.

```julia
# init application driven learning model
model = ADL.Model()
@variables(model, begin
    x[i=1:2] >= 0, ADL.Policy
    d[i=1:2], ADL.Forecast
end)

# plan model
@variables(Plan(model), begin
    y_plan[i=1:2] >= 0
    w_plan[i=1:2] >= 0
end)
@constraints(Plan(model), begin
    [i=1:2], y_plan[i] + w_plan[i] <= x[i].plan
    [i=1:2], y_plan[i] <= d[i].plan
end)
@objective(Plan(model), Min, sum(c[i] * x[i].plan - q[i] * y_plan[i] - r[i] * w_plan[i] for i in 1:2))

# assess model
@variables(Assess(model), begin
    y_assess[i=1:2] >= 0
    w_assess[i=1:2] >= 0
end)
@constraints(Assess(model), begin
    [i=1:2], y_assess[i] + w_assess[i] <= x[i].assess
    [i=1:2], y_assess[i] <= d[i].assess
end)
@objective(Assess(model), Min, sum(c[i] * x[i].assess - q[i] * y_assess[i] - r[i] * w_assess[i] for i in 1:2))

set_optimizer(model, Gurobi.Optimizer)
set_silent(model)

# forecast model
pred = Flux.Dense(1 => 2, exp)
ADL.set_forecast_model(model, pred)
```

Then, we can initialize the data, referencing forecast variables.

```julia
x_d = ones(100, 1) .|> Float32
y_d = Dict(
    d[1] => rand(10:100, 100) .|> Float32,
    d[2] => rand(10:100, 100) .|> Float32
)
```

We can check how the model performs by computing the assess cost with the initial (random) forecast model.

```julia
julia> ADL.compute_cost(model, x_d, y_d)
-3.571615f0
```

Now let's train the model using the GradientMode.

```julia
julia> gd_sol = ApplicationDrivenLearning.train!(
    model, x_d, y_d,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.GradientMode,
        rule=Flux.Adam(0.1),
        epochs=30
    )
)
Epoch 1 | Time = 0.4s | Cost = -3.57
Epoch 2 | Time = 0.8s | Cost = -4.36
Epoch 3 | Time = 1.2s | Cost = -5.33
Epoch 4 | Time = 1.6s | Cost = -6.51
Epoch 5 | Time = 2.0s | Cost = -7.95
Epoch 6 | Time = 2.4s | Cost = -9.72
Epoch 7 | Time = 2.8s | Cost = -11.88
Epoch 8 | Time = 3.2s | Cost = -14.53
Epoch 9 | Time = 3.6s | Cost = -17.78
Epoch 10 | Time = 4.0s | Cost = -21.77
Epoch 11 | Time = 4.4s | Cost = -26.68
Epoch 12 | Time = 4.8s | Cost = -32.73
Epoch 13 | Time = 5.2s | Cost = -39.52
Epoch 14 | Time = 5.6s | Cost = -46.64
Epoch 15 | Time = 6.0s | Cost = -54.15
Epoch 16 | Time = 6.4s | Cost = -62.95
Epoch 17 | Time = 6.8s | Cost = -74.74
Epoch 18 | Time = 7.2s | Cost = -90.36
Epoch 19 | Time = 7.6s | Cost = -110.35
Epoch 20 | Time = 8.0s | Cost = -135.12
Epoch 21 | Time = 8.4s | Cost = -164.34
Epoch 22 | Time = 8.8s | Cost = -197.82
Epoch 23 | Time = 9.2s | Cost = -237.1
Epoch 24 | Time = 9.6s | Cost = -282.57
Epoch 25 | Time = 10.0s | Cost = -334.87
Epoch 26 | Time = 10.4s | Cost = -389.66
Epoch 27 | Time = 10.8s | Cost = -442.7
Epoch 28 | Time = 11.2s | Cost = -469.92
Epoch 29 | Time = 11.6s | Cost = -452.58
Epoch 30 | Time = 12.0s | Cost = -430.85
ApplicationDrivenLearning.Solution(-469.91516f0, Real[1.6040976f0, 1.2566354f0, 2.8811285f0, 1.1966338f0])

julia> ADL.compute_cost(model, x_d, y_d)
-469.91516f0
```

After training, we can check the cost of the solution found by the gradient mode and even analyze the predictions from it.

```julia
julia> model.forecast(x_d[1,:])
2-element ApplicationDrivenLearning.VariableIndexedVector{Float32}:
 88.69701
 11.626293
```

As we can see, the forecast model overestimates the demand for the first item and underestimates the demand for the second item (both items average demand is 55), following the incentives from the model structure.