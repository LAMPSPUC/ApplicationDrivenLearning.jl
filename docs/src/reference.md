# [API](@id API)

This section documents the ApplicationDrivenLearning API.

## Constructors

```@docs
Model
PredictiveModel
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
ApplicationDrivenLearning.NelderMeadMode
ApplicationDrivenLearning.GradientMode
ApplicationDrivenLearning.NelderMeadMPIMode
ApplicationDrivenLearning.GradientMPIMode
ApplicationDrivenLearning.BilevelMode
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
```

### Flux attributes getters and setters

```@docs
ApplicationDrivenLearning.extract_flux_params
ApplicationDrivenLearning.fix_flux_params_single_model
ApplicationDrivenLearning.fix_flux_params_multi_model
ApplicationDrivenLearning.has_params
ApplicationDrivenLearning.apply_gradient!
```

## Other functions

```@docs
compute_cost
train!
ApplicationDrivenLearning.build_plan_model_forecast_params
ApplicationDrivenLearning.build_assess_model_policy_constraint
ApplicationDrivenLearning.build
ApplicationDrivenLearning.dict_to_var_indexed_matrix
ApplicationDrivenLearning.compute_single_step_cost
ApplicationDrivenLearning.compute_single_step_gradient
```

## Training algorithms

These are called by [`train!`](@ref) according to the selected mode and are
not meant to be called directly.

```@docs
ApplicationDrivenLearning.train_with_nelder_mead!
ApplicationDrivenLearning.train_with_gradient!
ApplicationDrivenLearning.train_with_nelder_mead_mpi!
ApplicationDrivenLearning.train_with_gradient_mpi!
ApplicationDrivenLearning.solve_bilevel
```