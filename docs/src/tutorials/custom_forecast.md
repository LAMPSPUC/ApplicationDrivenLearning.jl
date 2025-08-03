# Custom forecast models

The basic approach to define a forecast model is to use a `Chain` from the `Flux.jl` package, that directly maps the input to the output. But there are cases where this approach is not enough.

## Input-Output mapping

The connection between predictive model outputs and plan model inputs is not always a straightforward one. Because of this, the `set_forecast_model` function, used to define the predictive model in the ApplicationDrivenLearning.jl package, includes the `input_output_map` parameter.
This parameter allows users to declare an explicit mapping between the outputs produced by Flux models and the forecast variables used in the planning model. This is useful in contexts where the same prediction logic can be applied across several entities (such as production units or geographical locations), promoting model reuse and computational efficiency.

Consider a scenario where the input dataset contains 3 predictive variables (for example expected temperature on location 1, expected temperature on location 2 and weekday), there are 2 forecast variables (energy demand on the two locations of interest) and the forecast model should use only the expected temperature of a location to predict it’s demand. That means we would make two predictions using the same model and concatenate those values. This can be easily achieved with a dictionary mapping the data input and forecast variable indexes.

```julia
X = [
    76 89 2;
    72 85 3
] # input dataset of size 2 by 3
dem_forecast = Dense(
    2 => 1
) # forecast model takes 2 inputs and outputs single value

input_output_map = Dict(
    [1, 3] => [1], # input indexes 1 and 3 map to 1st forecast variable
    [2, 3] => [2] # input indexes 2 and 3 map to 2nd forecast variable
)
ApplicationDrivenLearning.set_forecast_model(model, dem_forecast, input_output_map)
```

## Multiple Flux models

The definition of the predictive model can also be done using multiple Flux models. This supports the modular construction of predictive architectures, where specialized components are trained to forecast different aspects of the problem, without the difficulty of defining custom architectures.

This can be achieved providing an array of model objects and an array of dictionaries as input-output mapping to the `set_forecast_model` function. Using the context from previous example, let’s assume there is an additional variable that has to be predicted to each location but not variable on time (that is, on dataset samples). This can be achieved defining an additional model that maps a constant input value to the correct output indexes.

```julia
X = [
    76 89 2 1;
    72 85 3 1
] # input dataset of size 2 by 4
dem_forecast = Dense(
    2 => 1
) # demand forecast model takes 2 inputs and outputs single value
aux_forecast = Dense(
    1 => 2
) # auxiliar forecast model takes 1 input and outputs 2 values
forecast_objs = [dem_forecast, aux_forecast]
input_output_map = [
    Dict(
        [1, 3] => [1],
        [2, 3] => [2]
    ), # input indexes 1,2,3 are used to compute forecast vars 1,2 with 1st Flux.Dense object
    Dict(
        [4] => [3, 4]
    ), # input index 4 is used to compute forecast vars 3,4 with 2nd Flux.Dense object
]
ApplicationDrivenLearning.set_forecast_model(model, forecast_objs, input_output_map)
```
