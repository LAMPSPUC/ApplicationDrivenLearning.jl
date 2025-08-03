# ApplicationDrivenLearning.jl Documentation

## Introduction

ApplicationDrivenLearning.jl is a Julia package for training time series forecast models using the application driven learning framework, that connects the optimization problem final cost with predictive model parameters in order to achieve the best model for a given application.

## Installation

Install ApplicationDrivenLearning with Julia's built-in package manager:

```julia
julia> import Pkg

julia> Pkg.add("ApplicationDrivenLearning")
```

For properly modelling plan, assess and predictive models, you will also need `JuMP` and `Flux` packages.

```julia
julia> Pkg.add("JuMP")
julia> Pkg.add("Flux")
```

Finally, for actually training the model, you will need to be able to solve the optimization models and the appropriate package depending on the training mode.

## Getting started

- Learn the basics of [JuMP](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_JuMP/) and [Julia](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/getting_started_with_julia/) in the [JuMP documentation](https://jump.dev/JuMP.jl/stable/)
- Follow the tutorials in this manual

If you need help, please open a GitHub issue.

## License

ApplicationDrivenLearning.jl is licensed under the [MIT License](https://github.com/LAMPSPUC/ApplicationDrivenLearning.jl/blob/main/LICENSE).

## Contributing

* PRs such as adding new training modes and fixing bugs are very welcome!
* For nontrivial changes, you'll probably want to first discuss the changes via issue.