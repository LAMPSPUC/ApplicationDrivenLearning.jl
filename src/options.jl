abstract type AbstractOptimizationMode end

"""
    BilevelMode <: AbstractOptimizationMode

Used to solve the application driven learning training problem as a bilevel
optimization problem by using the BilevelJuMP.jl package.

...

# Parameters

  - `optimizer::Function` is equivalent to `solver` in BilevelJuMP.BilevelModel.
  - `silent::Bool` is equivalent to `silent` in BilevelJuMP.BilevelModel.
  - `mode::Union{Nothing, BilevelJuMP.BilevelMode}` is equivalent to `mode` in BilevelJuMP.BilevelModel.
    ...
"""
struct BilevelMode <: AbstractOptimizationMode end

"""
    NelderMeadMode <: AbstractOptimizationMode

Used to solve the application driven learning training problem using the
Nelder-Mead optimization method implementation from Optim.jl package.

...

# Parameters

  - `initial_simplex` is the initial simplex of solutions to be applied.
  - `parameters` is the parameters to be applied to the Nelder-Mead optimization method.
    ...
"""
struct NelderMeadMode <: AbstractOptimizationMode end

"""
    GradientMode <: AbstractOptimizationMode

Used to solve the application driven learning training problem using the
gradient optimization method

...

# Parameters

  - `rule` is the Flux/Optimisers rule to be used in the gradient optimization
    process. Defaults to `Flux.Descent()`.
  - `epochs` is the number of epochs to be used in the gradient optimization
    process. Defaults to `100`.
  - `batch_size` is the batch size to be used in the gradient optimization
    process. Defaults to `-1`, meaning the full dataset is used at every epoch.
    When positive, each epoch draws `batch_size` sample indexes uniformly *with
    replacement*, so a batch may repeat samples and an epoch does not sweep the
    whole dataset.
  - `verbose` is the flag of whether to print the training process. Defaults to
    `true`.
  - `compute_cost_every` is the epoch frequency for computing the cost and
    evaluating best solution. Defaults to `1`.
  - `time_limit` is the time limit for the training process, in seconds.
    Defaults to `Inf`.
  - `g_tol` is the tolerance on the infinity norm of the cost gradients with
    respect to the forecasts, below which training stops. Defaults to `0`,
    which disables the check.
    ...
"""
struct GradientMode <: AbstractOptimizationMode end

"""
    NelderMeadMPIMode <: AbstractOptimizationMode

MPI implementation of [`NelderMeadMode`](@ref), which distributes the
per-sample cost evaluations across MPI processes.

...

# Parameters

  - `mpi_finalize::Bool` controls whether `MPI.Finalize()` is called at the end
    of training. Defaults to `true`.
  - Any other parameter accepted by `Optim.Options`. Unlike
    [`NelderMeadMode`](@ref), `initial_simplex` and `parameters` are not
    supported.
    ...
"""
struct NelderMeadMPIMode <: AbstractOptimizationMode end

"""
    GradientMPIMode <: AbstractOptimizationMode

MPI implementation of [`GradientMode`](@ref), which distributes the per-sample
cost and gradient evaluations across MPI processes.

...

# Parameters

  - The same parameters as [`GradientMode`](@ref).
  - `mpi_finalize::Bool` controls whether `MPI.Finalize()` is called at the end
    of training. Defaults to `true`.
    ...
"""
struct GradientMPIMode <: AbstractOptimizationMode end

"""
    Options(mode; params...)

Options struct to hold optimization mode and mode parameters.

`mode` must be a subtype of `AbstractOptimizationMode` (the type itself, not
an instance); the accepted keyword arguments are documented on each mode.

...

# Example

```julia
options = Options(
    GradientMode;
    rule = Flux.RMSProp(0.01),
    epochs = 100,
    batch_size = 10,
)
```

...
"""
struct Options
    mode::Any
    params::Dict{Symbol,Any}

    function Options(mode; params...)
        if mode isa Type && mode <: AbstractOptimizationMode
            return new(mode, Dict(params))
        else
            throw(
                ArgumentError(
                    "mode must be a subtype of AbstractOptimizationMode",
                ),
            )
        end
    end
end
