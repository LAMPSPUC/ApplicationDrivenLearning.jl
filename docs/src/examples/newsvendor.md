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

Let's start by loading the necessary packages and defining the data.

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
y_d = rand(10:100, (100, 2)) .|> Float32
x_d = ones(100, 1) .|> Float32
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

We can check how the model performs by computing the assess cost with the initial (random) forecast model.

```julia
julia> ADL.compute_cost(model, x_d, y_d)
-5.118482679128647
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
Epoch 1 | Time = 0.5s | Cost = -5.12
Epoch 2 | Time = 1.0s | Cost = -6.25
Epoch 3 | Time = 1.5s | Cost = -7.64
Epoch 4 | Time = 2.1s | Cost = -9.33
Epoch 5 | Time = 2.6s | Cost = -11.4
Epoch 6 | Time = 3.1s | Cost = -13.93
Epoch 7 | Time = 3.6s | Cost = -17.02
Epoch 8 | Time = 4.1s | Cost = -20.82
Epoch 9 | Time = 4.7s | Cost = -25.17
Epoch 10 | Time = 5.2s | Cost = -29.51
Epoch 11 | Time = 5.7s | Cost = -33.42
Epoch 12 | Time = 6.3s | Cost = -37.56
Epoch 13 | Time = 6.8s | Cost = -42.92
Epoch 14 | Time = 7.5s | Cost = -50.45
Epoch 15 | Time = 8.2s | Cost = -60.3
Epoch 16 | Time = 8.9s | Cost = -72.4
Epoch 17 | Time = 9.5s | Cost = -87.18
Epoch 18 | Time = 10.2s | Cost = -105.31
Epoch 19 | Time = 10.8s | Cost = -127.53
Epoch 20 | Time = 11.4s | Cost = -154.36
Epoch 21 | Time = 12.1s | Cost = -185.68
Epoch 22 | Time = 12.8s | Cost = -222.14
Epoch 23 | Time = 13.4s | Cost = -265.19
Epoch 24 | Time = 14.0s | Cost = -315.45
Epoch 25 | Time = 14.5s | Cost = -370.01
Epoch 26 | Time = 15.1s | Cost = -425.62
Epoch 27 | Time = 15.7s | Cost = -464.52
Epoch 28 | Time = 16.3s | Cost = -461.25
Epoch 29 | Time = 16.9s | Cost = -439.36
Epoch 30 | Time = 17.5s | Cost = -419.52
ApplicationDrivenLearning.Solution(-464.5160680770874, Real[1.6317965f0, 1.7067692f0, 2.7623773f0, 0.9124785f0])

julia> ADL.compute_cost(model, x_d, y_d)
-464.5160680770874
```

After training, we can check the cost of the solution found by the gradient mode and even analyze the predictions from it.

```julia
julia> model.forecast(x_d[1,:])
2-element Vector{Float32}:
 80.977684
 13.725394
```

As we can see, the forecast model overestimates the demand for the first item and underestimates the demand for the second item (both items average demand is 55), following the incentives from the model structure.