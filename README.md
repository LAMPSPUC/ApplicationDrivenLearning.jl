# ApplicationDrivenLearning.jl

ApplicationDrivenLearning.jl is a Julia package for training time series models using the application driven learning framework, that connects the optimization problem final cost with predictive model parameters in order to achieve the best model for a given application.

## Usage

```julia
import Pkg

Pkg.add("ApplicationDrivenLearning")  # not working yet! clone the repo instead

using ApplicationDrivenLearning

## Single power plan problem

# data
X = reshape([1 1], (2, 1))
Y = reshape([0 2], (2, 1))

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
    c1 ≥ 0
    c2 ≥ 0
end)
@constraints(ApplicationDrivenLearning.Assess(model), begin
    c1 ≥ 100 * (θ.assess-z.assess)
    c2 ≥ 20 * (z.assess-θ.assess)
end)
@objective(ApplicationDrivenLearning.Assess(model), Min, 10*z.assess + c1 + c2)

# basic setting
set_optimizer(model, HiGHS.Optimizer)
set_silent(model)

# forecast model
nn = Chain(Dense(1 => 1; bias=false))
ApplicationDrivenLearning.set_forecast_model(model, nn)

# training and getting solution
solution = ApplicationDrivenLearning.train!(
    model,
    X,
    Y,
    ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.NelderMeadMode
    )
)
print(solution.params)
```

## Installation

This package is **not yet** registered so if you want to use or test the code clone this repo and include source code from `src` directory.

## Contributing

* PRs such as adding new models and fixing bugs are very welcome!
* For nontrivial changes, you'll probably want to first discuss the changes via issue.