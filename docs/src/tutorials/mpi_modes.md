# MPI training modes

`NelderMeadMPIMode` and `GradientMPIMode` are distributed counterparts of
[`NelderMeadMode` and `GradientMode`](modes.md). The optimization algorithm is
unchanged; what is distributed is the *per-sample* work, i.e. solving the plan
and assess models (and, for the gradient mode, differentiating the plan model)
for each row of the dataset.

This is worthwhile when the plan/assess models are expensive relative to the
predictive model, which is the usual case: cost evaluation is `O(T)` solver
calls per iteration and those calls are independent.

## How it works

Both modes use [JobQueueMPI.jl](https://github.com/psrenergy/JobQueueMPI.jl).
The rank 0 process acts as the *controller*: it runs the optimization
algorithm, holds the incumbent parameters and dispatches one job per sample.
All other ranks are *workers*: they receive a parameter vector and a sample
index, apply the parameters to the forecast model, solve the plan and assess
models for that sample, and send the cost (and gradient) back.

Because every rank builds the full `ApplicationDrivenLearning.Model`, the model
definition script must be executed by all ranks — only the `train!` call
behaves differently depending on the rank.

## Running a script

MPI modes are not usable from an interactive session; the script has to be
launched under an MPI runner with more than one process:

```
$ mpiexec -n 4 julia --project my_training_script.jl
```

With `-n 4` there is one controller and three workers.

## NelderMeadMPIMode

### Arguments

  - `mpi_finalize`: whether `MPI.Finalize()` is called at the end of `train!`.
    Defaults to `true`. Set it to `false` when you want to run several `train!`
    calls (or other MPI work) in the same process.
  - Any other argument accepted by `Optim.Options`, such as `iterations`,
    `time_limit` and `g_abstol`, can be passed directly.

Unlike [`NelderMeadMode`](modes.md#Nelder-Mead-mode), this mode does not accept
the `initial_simplex` and `parameters` arguments; the default `Optim.NelderMead`
configuration is always used.

### Example

```julia
opt = ApplicationDrivenLearning.Options(
    ApplicationDrivenLearning.NelderMeadMPIMode,
    iterations = 100,
)
sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
```

## GradientMPIMode

### Arguments

The arguments are the same as [`GradientMode`](modes.md#Gradient-mode) —
`rule`, `epochs`, `batch_size`, `verbose`, `compute_cost_every`, `time_limit`
and `g_tol` — plus:

  - `mpi_finalize`: whether `MPI.Finalize()` is called at the end of `train!`.
    Defaults to `true`.

### Example

```julia
opt = ApplicationDrivenLearning.Options(
    ApplicationDrivenLearning.GradientMPIMode,
    rule = Flux.Adam(0.1),
    epochs = 30,
)
sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
```

## Return value

Only the controller process returns a meaningful [`Solution`](@ref
ApplicationDrivenLearning.Solution); worker processes return a placeholder and
their forecast model is left in an unspecified state. Guard any post-processing
accordingly.

!!! warning
    `mpi_finalize` defaults to `true`, and once MPI is finalized no MPI routine
    may be called again — including `JQM.is_controller_process()`. So either
    capture the rank *before* training:

    ```julia
    import JobQueueMPI as JQM

    JQM.mpi_init()
    is_controller = JQM.is_controller_process()

    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    if is_controller
        println(sol.cost)
    end
    ```

    or keep MPI alive until the post-processing is done:

    ```julia
    opt = ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.GradientMPIMode;
        epochs = 30,
        mpi_finalize = false,
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    if JQM.is_controller_process()
        println(sol.cost)
    end
    JQM.mpi_finalize()
    ```

The second form is also what you need when running several `train!` calls in
the same process: every call but the last must pass `mpi_finalize = false`.
