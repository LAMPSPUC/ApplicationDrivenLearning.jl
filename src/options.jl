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
    OptimMode <: AbstractOptimizationMode

Used to solve the application driven learning training problem with any
algorithm from the Optim.jl package.

The objective handed to Optim is the assessed cost as a function of the flat
parameter vector of the predictive model. Derivative-free algorithms need nothing
further; gradient-based ones are given `dC/dθ` automatically, obtained by
carrying the same reverse pass the [`GradientMode`](@ref) loop uses one step
further, from the forecasts to the parameters.

...

# Parameters

  - `algorithm` is the Optim.jl algorithm object to use. Defaults to
    `Optim.NelderMead()`. Anything Optim provides works, for example
    `Optim.ParticleSwarm()`, `Optim.SimulatedAnnealing()`, `Optim.LBFGS()`,
    `Optim.BFGS()` or `Optim.ConjugateGradient()`.
  - `parallel` is the parallel backend used for the per-sample cost and gradient
    evaluations. Defaults to `SerialBackend()`; pass `MPIBackend()` to
    distribute them across MPI processes, or
    `DistributedBackend()` across `Distributed` worker
    processes. See [`AbstractParallelBackend`](@ref).
  - Any other parameter accepted by `Optim.Options`, such as `iterations`,
    `time_limit`, `show_trace` or `g_abstol`.
    ...

# Example

```julia
options =
    Options(OptimMode; algorithm = Optim.ParticleSwarm(), iterations = 500)
```
"""
struct OptimMode <: AbstractOptimizationMode end

"""
    NLoptMode <: AbstractOptimizationMode

Used to solve the application driven learning training problem with an algorithm
from the NLopt.jl package.

!!! note

    Provided by a package extension: **`using NLopt` is required** before
    training with this mode. The mode type itself always exists, so that a
    missing `using` produces a readable error rather than an `UndefVarError`.

...

# Parameters

  - `algorithm::Symbol` is the NLopt algorithm to use. Defaults to
    `:LN_NELDERMEAD`. The derivative-free families are `:LN_*` (local, e.g.
    `:LN_BOBYQA`, `:LN_COBYLA`) and `:GN_*` (global, e.g. `:GN_DIRECT`); the
    `:LD_*` family is gradient-based and is given `dC/dθ` automatically.
  - Any other parameter is set on the NLopt `Opt` object, so the stopping
    criteria are named as NLopt names them: `xtol_rel`, `xtol_abs`, `ftol_rel`,
    `ftol_abs`, `maxeval`, `maxtime`, `stopval`. Note this differs from
    [`OptimMode`](@ref), where such keywords go to `Optim.Options` instead.
  - `parallel` is the parallel backend used for the per-sample cost and gradient
    evaluations. Defaults to `SerialBackend()`; pass `MPIBackend()` to
    distribute them across MPI processes, or
    `DistributedBackend()` across `Distributed` worker
    processes. See [`AbstractParallelBackend`](@ref).

...

# Example

```julia
using NLopt

options =
    Options(NLoptMode; algorithm = :LN_BOBYQA, xtol_rel = 1e-6, maxeval = 500)
```
"""
struct NLoptMode <: AbstractOptimizationMode end

"""
    NelderMeadMode <: AbstractOptimizationMode

Used to solve the application driven learning training problem using the
Nelder-Mead optimization method implementation from Optim.jl package.

!!! compat "Deprecated"

    Superseded by [`OptimMode`](@ref), which reaches every Optim.jl algorithm
    rather than just this one. `NelderMeadMode` behaves exactly as before and is
    kept so that existing code keeps working; new code should prefer

    ```julia
    Options(OptimMode; algorithm = Optim.NelderMead())
    ```

    Note that `initial_simplex` and `parameters` are Nelder-Mead specific, so
    under `OptimMode` they move into the algorithm object:
    `Optim.NelderMead(initial_simplex = ...)`.

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
  - `parallel` is the parallel backend used for the per-sample cost and gradient
    evaluations. Defaults to `SerialBackend()`; pass `MPIBackend()` to
    distribute them across MPI processes, or
    `DistributedBackend()` across `Distributed` worker
    processes. See [`AbstractParallelBackend`](@ref).
    ...
"""
struct GradientMode <: AbstractOptimizationMode end

"""
    NelderMeadMPIMode <: AbstractOptimizationMode

MPI implementation of [`NelderMeadMode`](@ref), which distributes the
per-sample cost evaluations across MPI processes.

!!! compat "Deprecated"

    Parallelism is now an option rather than a mode, so any optimizer can be
    run over MPI rather than just Nelder-Mead:

    ```julia
    Options(OptimMode; algorithm = Optim.NelderMead(), parallel = MPIBackend())
    ```

    This mode keeps working and forwards to exactly that, translating
    `mpi_finalize` onto the backend.

...

# Parameters

  - `mpi_finalize::Bool` controls whether `MPI.Finalize()` is called at the end
    of training. Defaults to `true`.
  - Any other parameter accepted by `Optim.Options`.
    ...
"""
struct NelderMeadMPIMode <: AbstractOptimizationMode end

"""
    GradientMPIMode <: AbstractOptimizationMode

MPI implementation of [`GradientMode`](@ref), which distributes the per-sample
cost and gradient evaluations across MPI processes.

!!! compat "Deprecated"

    Parallelism is now an option rather than a mode:

    ```julia
    Options(GradientMode; parallel = MPIBackend())
    ```

    This mode keeps working and forwards to exactly that, translating
    `mpi_finalize` onto the backend.

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
