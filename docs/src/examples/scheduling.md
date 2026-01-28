# Minimal Scheduling Problem

This example shows how to use the ApplicationDrivenLearning.jl package to solve a minimal scheduling problem.

In this problem, we have to define the dispatch of a single unit, considering it's operational constraints, costs and the demand. The cost for under-dispatching is 100 and the cost for over-dispatching is 20. The forecast model will be a simple linear model.

The plan model will only assign the dispatch of the unit equal to the forecast. The assess model will apply a correction to the dispatch, considering the under-dispatching and over-dispatching costs.

First, let's load the necessary packages.

```julia
using Flux
using JuMP
using Gurobi
using ApplicationDrivenLearning

ADL = ApplicationDrivenLearning
```

Now, we can initialize the application driven learning model, build the plan and assess models and set the forecast model.

```julia
# init application driven learning model
model = ADL.Model()
@variable(model, z >= 0, ADL.Policy)
@variable(model, y, ADL.Forecast)

# plan model
@constraints(ADL.Plan(model), begin
    z.plan == y.plan
end)
@objective(ADL.Plan(model), Min, 10*z.plan)

# assess model
@variables(ADL.Assess(model), begin
    correction
    under_dispatch >= 0
    over_dispatch >= 0
end)
@constraints(ADL.Assess(model), begin
    correction == y.assess - z.assess
    under_dispatch >= correction
    over_dispatch >= -correction
end)
@objective(ADL.Assess(model), Min, 10*y.assess + 100*under_dispatch + 20*over_dispatch)

# forecast model
predictive = Dense(1 => 1, exp; bias=false)
ADL.set_forecast_model(model, predictive)
```

We can check how the model performs by computing the assess cost with the initial (random) forecast model.

```julia
X = ones(2, 1) .|> Float32
Y = Dict(y => [0.0, 2.0] .|> Float32)
set_optimizer(model, Gurobi.Optimizer)
set_silent(model)
```

```julia
julia> ADL.compute_cost(model, X, Y)
99.7323203086853
```

And finally, we can train the model using the Nelder-Mead mode.

```julia
julia> solution = ADL.train!(
    model, X, Y,
    ADL.Options(
        ADL.NelderMeadMode,
        iterations=100,
        show_trace=true
    )
)
Iter     Function value    √(Σ(yᵢ-ȳ)²)/n 
------   --------------    --------------
     0     9.973232e+01     2.466946e+00
 * time: 0.0
     1     9.973232e+01     3.148899e+01
 * time: 0.006000041961669922
     2     3.675434e+01     1.421373e+01
 * time: 0.014999866485595703
     3     3.675434e+01     2.673199e+00
 * time: 0.0279998779296875
     4     3.140795e+01     6.259584e-01
 * time: 0.039999961853027344
     5     3.015603e+01     1.546907e-01
 * time: 0.04699993133544922
     6     3.015603e+01     3.849030e-02
 * time: 0.05299997329711914
     7     3.007905e+01     3.841400e-02
 * time: 0.05799984931945801
     8     3.000222e+01     9.596825e-03
 * time: 0.06699991226196289
     9     3.000222e+01     2.398491e-03
 * time: 0.07599997520446777
    10     3.000222e+01     5.979546e-04
 * time: 0.0839998722076416
    11     3.000103e+01     3.366484e-04
 * time: 0.08899998664855957
    12     3.000035e+01     1.144409e-04
 * time: 0.09500002861022949
    13     3.000012e+01     3.814697e-05
 * time: 0.10699987411499023
    14     3.000005e+01     9.536743e-06
 * time: 0.11500000953674316
    15     3.000003e+01     9.536743e-06
 * time: 0.12199997901916504
    16     3.000001e+01     9.536743e-06
 * time: 0.1269998550415039
    17     2.999999e+01     3.015783e-06
 * time: 0.13899993896484375
    18     2.999999e+01     3.015783e-06
 * time: 0.14399981498718262
    19     2.999998e+01     1.907349e-06
 * time: 0.15400004386901855
    20     2.999998e+01     0.000000e+00
 * time: 0.1640000343322754
ApplicationDrivenLearning.Solution(29.99998f0, Real[0.6931467f0])

julia> model.forecast(X[1,:])  # final forecast
1-element Vector{Float32}:
 1.999999

julia> ADL.compute_cost(model, X, Y)  # final cost
29.99998
```