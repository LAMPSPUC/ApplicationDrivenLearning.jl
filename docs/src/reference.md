# [API](@id API)

This section documents the public ApplicationDrivenLearning API.

Functions and types whose name starts with an underscore are internal
implementation details: they are not part of the public API, are not listed
here, and may change without a breaking release.

## Constructors

```@docs
Model
ForecastModel
FullForecastModel
Plan
Assess
```

## JuMP variable types

```@docs
Policy
Forecast
```

## Structs

```@docs
ApplicationDrivenLearning.Options
ApplicationDrivenLearning.Solution
```

## Modes

```@docs
ApplicationDrivenLearning.OptimMode
ApplicationDrivenLearning.NLoptMode
ApplicationDrivenLearning.GradientMode
ApplicationDrivenLearning.BilevelMode
```
## Deprecated Modes

```@docs
ApplicationDrivenLearning.NelderMeadMode
ApplicationDrivenLearning.NelderMeadMPIMode
ApplicationDrivenLearning.GradientMPIMode
```

## Parallel backends

```@docs
ApplicationDrivenLearning.AbstractParallelBackend
ApplicationDrivenLearning.SerialBackend
ApplicationDrivenLearning.MPIBackend
ApplicationDrivenLearning.DistributedBackend
```

## Attributes getters and setters

```@docs
ApplicationDrivenLearning.plan_policy_vars
ApplicationDrivenLearning.assess_policy_vars
ApplicationDrivenLearning.plan_forecast_vars
ApplicationDrivenLearning.assess_forecast_vars
ApplicationDrivenLearning.set_forecast_model
ApplicationDrivenLearning.extract_params
ApplicationDrivenLearning.apply_params
ApplicationDrivenLearning.apply_gradient!
```

## Other functions

```@docs
compute_cost
train!
```
