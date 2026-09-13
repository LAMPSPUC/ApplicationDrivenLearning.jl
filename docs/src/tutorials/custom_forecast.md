# Custom forecast models

The basic approach to define a forecast model is to use a `Chain` from the `Flux.jl` package, that directly maps the input to the output. But there are cases where this approach is not enough.

A forecast is described by [`ForecastModel`](@ref) units. Each unit is one architecture together with the input columns it reads and the forecast variables it predicts:

```julia
ForecastModel(
    inputs = [:temp, :hour],     # column positions or column names
    architecture = Dense(2 => 1),
    outputs = [demand],          # the Forecast variables this architecture predicts
)
```

A vector of units goes to [`set_forecast_model`](@ref), which checks that between them they predict every forecast variable declared on the model, exactly once.

## Wiring inputs to forecast variables

The connection between predictive model outputs and plan model inputs is not always a straightforward one. Writing one unit per architecture is what makes it explicit: a unit says which columns it reads, so the same prediction logic can be applied across several entities (such as production units or geographical locations) by declaring it once per entity.

Consider a scenario where the input dataset contains 3 variables (for example expected temperature on location 1, expected temperature on location 2 and weekday), there are 2 forecast variables (energy demand on the two locations of interest) and the forecast model should use only the expected temperature of a location to predict its demand. That is two units of the same shape, one per location.

```julia
model = ApplicationDrivenLearning.Model()
@variable(model, demand[1:2], ApplicationDrivenLearning.Forecast)

X = [
    76 89 2;
    72 85 3
] .|> Float32 # input dataset of size 2 by 3
Y = Dict(
    demand[1] => [101, 89] .|> Float32,
    demand[2] => [68, 49] .|> Float32
)

# the architecture each location uses: 2 inputs, one output value
dem_forecast = Dense(2 => 1)

ApplicationDrivenLearning.set_forecast_model(
    model,
    [
        ForecastModel(
            inputs = [1, 3],              # temperature at location 1, weekday
            architecture = dem_forecast,
            outputs = [demand[1]],
        ),
        ForecastModel(
            inputs = [2, 3],              # temperature at location 2, weekday
            architecture = deepcopy(dem_forecast),
            outputs = [demand[2]],
        ),
    ],
)
```

Note the `deepcopy`. Each unit trains **its own** parameters, so reusing an architecture across units means reusing its *shape*, not its weights — which is what "the same model for each location" almost always means, and what lets the two locations learn different sensitivities to temperature. Handing the very same object to two units is rejected rather than quietly tying their weights together:

```julia
# ERROR: Units 1 and 2 hold the same `architecture` object ...
ForecastModel(inputs = [1, 3], architecture = dem_forecast, outputs = [demand[1]]),
ForecastModel(inputs = [2, 3], architecture = dem_forecast, outputs = [demand[2]]),
```

A function returning a fresh network reads better when there are more than two:

```julia
dem_forecast() = Dense(2 => 1)
units = [
    ForecastModel(
        inputs = [i, 3],
        architecture = dem_forecast(),
        outputs = [demand[i]],
    ) for i in 1:2
]
```

### Naming the inputs

Integer indexes are positions in `X`, so `[1, 3]` only means "temperature at
location 1 and the weekday" as long as the columns arrive in that order. A unit's
`inputs` can be written as column names instead, which says what it means and
lets `X` be given as a table:

```julia
ApplicationDrivenLearning.set_forecast_model(
    model,
    [
        ForecastModel(
            inputs = [:temp_1, :weekday],
            architecture = dem_forecast(),
            outputs = [demand[1]],
        ),
        ForecastModel(
            inputs = [:temp_2, :weekday],
            architecture = dem_forecast(),
            outputs = [demand[2]],
        ),
    ],
)

using DataFrames
X = DataFrame(temp_1 = [76, 89], temp_2 = [72, 85], weekday = [2, 3])
```

The names become the forecast's input schema and are resolved to positions once, so
nothing else changes: a `DataFrame` may now hold its columns in any order, and any
column it is missing is reported by name. A matrix `X` is still read positionally,
in the order the units first mention the names — so writing the units in a
different order is how a different column order is asked for. There is no separate
place to declare the schema: naming a unit's `inputs` *is* the declaration.

Every unit has to select the same way, all by position or all by name, since there
is one input schema for the whole forecast. And no unit may read the same column
twice, which would feed two of its architecture's inputs from one column.

## Several architectures in one forecast

Units may hold different architectures, which supports the modular construction of predictive models where specialized components forecast different aspects of the problem, without the difficulty of defining custom architectures.

Using the context from the previous example, let's assume we also want to predict price for each location using a single model that receives average lagged price. That is one more unit, with its own architecture and its own outputs.

```julia
model = ApplicationDrivenLearning.Model()
@variables(model, begin
    demand[1:2], ApplicationDrivenLearning.Forecast
    price[1:2], ApplicationDrivenLearning.Forecast
end)

X = [
    76 89 2 103;
    72 85 3 89
] .|> Float32  # input dataset of size 2 by 4
Y = Dict(
    demand[1] => [101, 89] .|> Float32,
    demand[2] => [68, 49] .|> Float32,
    price[1] => [101, 89] .|> Float32,
    price[2] => [68, 49] .|> Float32,
)

ApplicationDrivenLearning.set_forecast_model(
    model,
    [
        # each demand variable from its own location's temperature and the weekday
        ForecastModel(
            inputs = [1, 3],
            architecture = Dense(2 => 1),
            outputs = [demand[1]],
        ),
        ForecastModel(
            inputs = [2, 3],
            architecture = Dense(2 => 1),
            outputs = [demand[2]],
        ),
        # and both prices from the lagged price, by one architecture with two
        # outputs - listed in the order it produces them
        ForecastModel(
            inputs = [4],
            architecture = Dense(1 => 2),
            outputs = [price[1], price[2]],
        ),
    ],
)
```

A unit with several `outputs` is the case where one architecture predicts several variables at once: `Dense(1 => 2)` produces two values from the lagged price, and `outputs` says which variable each row is. The number of outputs is checked against the architecture when the unit is built, so a `Dense(1 => 2)` listing three variables is an error there rather than a shape mismatch during training.

## Naming the columns of `Y`

Container declarations like `@variable(model, demand[1:2], Forecast)` name their variables `demand[1]` and `demand[2]`, which no data file is likely to carry. Write the `Y` column next to the variable it holds:

```julia
ForecastModel(
    inputs = [1, 3],
    architecture = Dense(2 => 1),
    outputs = [demand[1] => :demand_loc1],
)
```

The column name then travels with its variable, so there is no separate list to keep in the right order. A bare variable means "the column is named after the variable", which is what `@variable(model, demand, Forecast)` already gives you.

## Pinning the columns of a matrix `Y`

A name addresses a column of a named container — a table, a `DataFrame`, a `Dict` of series. A matrix has no names, so its columns are read in the order the `Forecast` variables were declared on the model. That works, but nothing states it and nothing checks it: insert a `@variable` above the others and every caller passing a matrix silently means something else.

Write the position instead, and the layout is declared:

```julia
ForecastModel(
    architecture = Dense(2 => 2),
    outputs = [demand[1] => 2, demand[2] => 1],   # Y column 2 holds demand[1]
)
```

Positions behave like the ones in `inputs`. They start at 1, no two variables may share one, and gaps are allowed — `outputs = [a => 1, b => 3]` requires a three-column `Y` and never looks at column 2. A `Y` too narrow to reach the widest declared column is an error rather than a guess.

The two spellings say the same thing, so all units of a forecast use one or the other, and a positional forecast refuses a *named* container: a table's columns have names, and matching those by order is exactly what this design refuses to do. Pass a matrix, or address the columns by name.
