# Getting started with ApplicationDrivenLearning

This is a quick introduction to modeling and training end-to-end forecast models with ApplicationDrivenLearning.

## A first example

We will train an energy load predictive model that is applied to a one-hour-ahead generation planning problem in a power system consisting of a single plant with the following characteristics: a production capacity of 4 MW and a generation cost of R\$10/MWh.

The available data is very limited: we don't have any auxiliary variable and just two samples of past demand.

Here is the complete code to model, train and extract the parameters of the predictive model:

```julia
using JuMP
using Flux
import HiGHS
using ApplicationDrivenLearning
using Optim

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
ApplicationDrivenLearning.set_forecast_model(model, ApplicationDrivenLearning.PredictiveModel(nn))

# training the full model
solution = ApplicationDrivenLearning.train!(
    model,
    X,
    Y,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.OptimMode;
        algorithm = Optim.NelderMead(),
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
using Optim
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

### Accepted input containers

`X` and `Y` are not restricted to matrices and dictionaries. Both
[`compute_cost`](@ref) and [`train!`](@ref) accept:

  - a matrix of size `(samples, features)`;
  - a vector, read as a single column;
  - any **Tables.jl**-compatible table — a `DataFrame`, a `NamedTuple` of
    vectors, a `CSV.File`, and so on;
  - for `Y` only, a `Dict` mapping each forecast variable to its series, as
    above.

The two arguments are independent, so a `DataFrame` for `X` and a `Dict` for `Y`
is fine. So the example above could equally be written:

```julia
using DataFrames
X = DataFrame(ones = Float32[1, 1])
Y = DataFrame(θ = Float32[10, 20])

# a table is read by its column names, so say which column is the input
ApplicationDrivenLearning.set_forecast_model(model, ApplicationDrivenLearning.PredictiveModel(nn; input_names = [:ones]))
```

#### How columns are matched

One rule covers both arguments:

> **A named container is matched by name; an unnamed container is matched by
> position. Neither is ever guessed at.**

An array carries no column names, so position is its only possible reading. A
table's columns *are* named, and those names are what gets used — so reordering
the columns of a `DataFrame` cannot change your results, and extra columns (an id
or a date, say) are ignored. If the names cannot be matched, you get an error
rather than a silent fallback to column order, which is how you would otherwise
end up training against the wrong series without any warning.

`Matrix(df)` is how you ask for positional matching, and reads as exactly that
request: it drops the names.

The names a table is matched against are the ones the model declares:

  - **`Y`** uses the names of the forecast variables. `@variable(model, θ,
    Forecast)` reads a `θ` column with nothing to configure. Container
    declarations like `@variable(model, θ[1:2], Forecast)` name their variables
    `θ[1]` and `θ[2]`, which no table is likely to carry, so pass
    `output_names = [:demand, :price]` to [`set_forecast_model`](@ref) — or use
    the `Dict` form, which is keyed by the variables themselves.
  - **`X`** uses `input_names`, either declared directly or implied by writing
    the `input_output_map` with `Symbol` keys:

```julia
# declared directly, for a single network applied to the whole input
ApplicationDrivenLearning.set_forecast_model(model, ApplicationDrivenLearning.PredictiveModel(Chain(Dense(2 => 1)); input_names = [:temp, :hour]))

# or implied, by naming the inputs in the map itself, where each network
# reads its own columns
PredictiveModel(
    [Dense(2 => 1), Dense(1 => 1)],
    [Dict([:temp, :hour] => [demand]), Dict([:price] => [spill])],
)
```

There is no exception for a table with a single column, which is why the
`DataFrame` version of this tutorial's data declares `input_names` above. Its
order cannot be wrong, but its *name* still can: a one-input model handed a
`humidity` column when it wanted `temp` is a mistake worth catching, and only a
declared schema catches it. `Y` needs nothing extra there, since
`@variable(model, θ, Forecast)` already names its own column.

#### How rows are matched

The same rule extends one dimension further. By default `X` and `Y` are lined up
by row order, and nothing checks that row `t` of one is the same observation as
row `t` of the other — only that they have the same number of rows. Naming the
column that identifies an observation fixes that:

```julia
ApplicationDrivenLearning.set_forecast_model(model, ApplicationDrivenLearning.PredictiveModel(nn; input_names = [:ones], sample_key = :timestamp))

X = DataFrame(timestamp = [10, 20, 30], ones = Float32[1, 1, 1])
Y = DataFrame(timestamp = [30, 10, 20], θ = Float32[30, 10, 20])  # any order
```

The realized values are then looked up by key for each row of `X`, so `Y` may
arrive in any order, and `Y` may hold samples that `X` does not ask for — the
row-wise counterpart of ignoring an unwanted column. A sample that `X` asks for
and `Y` lacks is an error rather than a silent off-by-one, and a key that repeats
is rejected because it cannot identify a sample.

`X` fixes the order, since that is the order the returned per-sample costs and
gradients are in. The key column is not a feature: it identifies a row, so it is
never fed to the network. And as on the column side, a container that carries no
row labels — a matrix, or the `Dict` form of `Y` — falls back to order.

Columns must be numeric. A column of strings, or one containing `missing`, is
rejected with an error naming the offending argument rather than failing later
inside the solver.


A simple forecast model with only one parameter can be defined as a `Flux.Dense` layer with just 1 weight and no bias. We can associate the predictive model with our ApplicationDrivenLearning model only if its output size matches the number of declared forecast variables.

```julia
nn = Chain(Dense(1 => 1; bias=false))
ApplicationDrivenLearning.set_forecast_model(model, ApplicationDrivenLearning.PredictiveModel(nn))
```

Finally, the full model is trained with [`OptimMode`](modes.md#Optim-mode), which reaches any algorithm from the `Optim` package — here Nelder-Mead. `Optim` is already a dependency of ApplicationDrivenLearning, so it does not need to be installed, but it does need `using Optim` to name the algorithm.

```julia
solution = ApplicationDrivenLearning.train!(
    model,
    X,
    Y,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.OptimMode;
        algorithm = Optim.NelderMead(),
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
