# Getting started with ApplicationDrivenLearning

This is a quick introduction modeling and training end-to-end forecast models with ApplicationDrivenLearning.

## A first example

We will train a energy load predictive model that is applied to a one-hour agead generation planning problem in a power system consisting of a single plant with the following characteristics: a production capacity of 4 MW and a generation cost of R$10/MWh.

The available data is very limited: we don't have any auxiliar variable and just two samples of past demand.

Here is the complete code to model, train and extract the parameters of the predictive model:

```julia
using JuMP
using Flux
import HiGHS
using ApplicationDrivenLearning

# main model and policy / forecast variables
model = ApplicationDrivenLearning.Model()
@variables(model, begin
    z, ApplicationDrivenLearning.Policy
    θ, ApplicationDrivenLearning.Forecast
end)

# plan model
@variables(ApplicationDrivenLearning.Plan(model), begin
    c1 ≥ 0
    c2 ≥ 0
end)
@constraints(ApplicationDrivenLearning.Plan(model), begin
    c1 ≥ 100 * (θ.plan-z.plan)
    c2 ≥ 20 * (z.plan-θ.plan)
end)
@objective(ApplicationDrivenLearning.Plan(model), Min, 10*z.plan + c1 + c2)

# assess model
@variables(ApplicationDrivenLearning.Assess(model), begin
    c3 ≥ 0
    c4 ≥ 0
end)
@constraints(ApplicationDrivenLearning.Assess(model), begin
    c3 ≥ 100 * (θ.assess-z.assess)
    c4 ≥ 20 * (z.assess-θ.assess)
end)
@objective(ApplicationDrivenLearning.Assess(model), Min, 10*z.assess + c3 + c4)

# basic setting
set_optimizer(model, HiGHS.Optimizer)
set_silent(model)

# data
X = reshape([1 1], (2, 1)) .|> Float32
Y = Dict(θ => [10, 20] .|> Float32)

# forecast model
nn = Chain(Dense(1 => 1; bias=false))
ApplicationDrivenLearning.set_forecast_model(model, nn)

# training the full model
solution = ApplicationDrivenLearning.train!(
    model,
    X,
    Y,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.NelderMeadMode
    )
)

# getting predictions
pred = model.forecast(X')

# extracting solution
println(solution.params)
```

## Step-by-step

Once installed, the necessary packages can be loaded into julia:

```julia
using JuMP
using Flux
using ApplicationDrivenLearning
```

We have to include a solver for solving the optimization models. In this case, we load HiGHS:
```julia
using HiGHS
```

Just like regular JuMP, ApplicationDrivenLearning has a `Model` function to initialize an empty model. After initializing, we can declare the policy and forecast variables.

- Policy variables represent decision variables that should be maintained from the `Plan` to the `Assess` model.
- Forecast variables represent future values relevant to the problem. They are replaced by forecast output values in the planning step and fixed as realized values (from the `Y` matrix) in the assessment step.

For our problem, the policy variable `z` represents the generation and forecast variable `θ` represents the demand.

```julia
model = ApplicationDrivenLearning.Model()
@variables(model, begin
    z, ApplicationDrivenLearning.Policy
    θ, ApplicationDrivenLearning.Forecast
end)
```

To populate the plan model, we follow a syntax very similar to the JuMP package, with the addition of a suffix on policy and forecast variables.

We declare additional variables `c1` and `c2` to model the overestimation and underestimation costs, that are added to the generation cost in the objective function. 

```julia
@variables(ApplicationDrivenLearning.Plan(model), begin
    c1 ≥ 0
    c2 ≥ 0
end)
@constraints(ApplicationDrivenLearning.Plan(model), begin
    c1 ≥ 100 * (θ.plan-z.plan)
    c2 ≥ 20 * (z.plan-θ.plan)
end)
@objective(ApplicationDrivenLearning.Plan(model), Min, 10*z.plan + c1 + c2)
```

The assess model can be declared in a similar way.

```julia
@variables(ApplicationDrivenLearning.Assess(model), begin
    c3 ≥ 0
    c4 ≥ 0
end)
@constraints(ApplicationDrivenLearning.Assess(model), begin
    c3 ≥ 100 * (θ.assess-z.assess)
    c4 ≥ 20 * (z.assess-θ.assess)
end)
@objective(ApplicationDrivenLearning.Assess(model), Min, 10*z.assess + c3 + c4)
```

We need to associate the model with an optimizer that can solve the plan and assess models. For this case, we use the HiGHS optimizer. We also set the model to silent mode to avoid excessive outputs from the solve iterations.

```julia
set_optimizer(model, HiGHS.Optimizer)
set_silent(model)
```

As explained, the data used to train the model is very limited, composed of only two samples of energy demand. Values of one are used as input data, without adding any real additional information to the model. `X` is a matrix representing input values and the dictionary `Y` maps the forecast variable `θ` to numerical values to be used. Both `X` and `Y` values are transformed to `Float32` type to match Flux parameters.

```julia
X = reshape([1 1], (2, 1)) .|> Float32
Y = Dict(θ => [10, 20] .|> Float32)
```


A simple forecast model with only one parameter can be defined as a `Flux.Dense` layer with just 1 weight and no bias. We can associate the predictive model with our ApplicationDrivenLearning model only if its output size matches the number of declared forecast variables.

```julia
nn = Chain(Dense(1 => 1; bias=false))
ApplicationDrivenLearning.set_forecast_model(model, nn)
```

Finally, the full model is trained using the `NelderMeadMode`. For using this mode, it is necessary to install the `Optim` package previously, but it doesn't need to be installed.

```julia
solution = ApplicationDrivenLearning.train!(
    model,
    X,
    Y,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.NelderMeadMode
    )
)
```

Now we can easily make new predictions with the trained model:

```julia
julia> pred = model.forecast(X')
1×2 Matrix{Float32}:
 20.0  20.0
```

Compute the assess cost for each data sample:
```julia
julia> ApplicationDrivenLearning.compute_cost(model, X, Y, false, false)
2-element Vector{Float64}:
 400.0
 200.0
```

And also extract the parameters from the trained model:

```julia
julia> solution.params
1-element Vector{Real}:
 20.0f0
```
