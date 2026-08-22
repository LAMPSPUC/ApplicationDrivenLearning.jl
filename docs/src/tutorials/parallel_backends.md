# Running in parallel

Parallelism is an **option**, not a mode. The `parallel` keyword can be given to
[`OptimMode`](modes.md#Optim-mode), [`NLoptMode`](modes.md#NLopt-mode) and
[`GradientMode`](modes.md#Gradient-mode) alike:

```julia
opt = ApplicationDrivenLearning.Options(
    ApplicationDrivenLearning.GradientMode,
    rule = Flux.Adam(0.1),
    epochs = 30,
    parallel = ApplicationDrivenLearning.MPIBackend(),
)
sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
```

The optimization algorithm is unchanged; what is distributed is the *per-sample*
work, i.e. solving the plan and assess models (and, when gradients are needed,
differentiating the plan model) for each row of the dataset. That is the only
thing distributed — the optimizer, the epoch loop and the parameter update all
run in the driver process — which is why the backend is independent of the mode.

[`BilevelMode`](modes.md#Bilevel-mode) is the exception: it builds a single MIP
over all samples and never evaluates a per-sample cost, so there is nothing to
distribute and it does not accept `parallel`.

This is worthwhile when the plan/assess models are expensive relative to the
predictive model, which is the usual case: cost evaluation is `O(T)` solver
calls per iteration and those calls are independent.

## Choosing a backend

|  | [`MPIBackend`](#Running-over-MPI) | [`DistributedBackend`](#Running-over-Distributed) |
| --- | --- | --- |
| Launching | `mpiexec -n N julia script.jl` | `addprocs(N)`, works from the REPL |
| Getting a model onto the workers | every rank runs the whole script | `train!(build_case, ...)` with a builder function |
| Return value | every rank returns; only the controller's is meaningful | only the driver runs `train!` at all |
| Requirements | a working MPI runtime | none; `Distributed` is a stdlib |
| Across machines | the usual HPC route, via the scheduler | `addprocs` with SSH |

Prefer `DistributedBackend` for interactive work and for a single machine —
there is nothing to install and no launcher to remember. Prefer `MPIBackend` on a
cluster whose scheduler already speaks MPI.

Both distribute exactly the same function, so switching between them changes
neither the result nor the optimizer. The tests assert this: each backend must
reproduce the serial cost *and* the serial parameters.

---

## Running over MPI

Uses [JobQueueMPI.jl](https://github.com/psrenergy/JobQueueMPI.jl). The rank 0
process acts as the *controller*: it runs the optimization algorithm, holds the
incumbent parameters and dispatches one job per sample. All other ranks are
*workers*: they receive a parameter vector and a sample index, apply the
parameters to the forecast model, solve the plan and assess models for that
sample, and send the cost (and gradient) back.

Because every rank builds the full `ApplicationDrivenLearning.Model`, the model
definition script must be executed by all ranks — only the `train!` call behaves
differently depending on the rank.

### Running a script

MPI is not usable from an interactive session; the script has to be launched
under an MPI runner with more than one process:

```
$ mpiexec -n 4 julia --project my_training_script.jl
```

With `-n 4` there is one controller and three workers.

### Arguments

Every mode keeps its own arguments unchanged; `parallel` is the only addition.
`MPIBackend` itself takes one:

  - `finalize`: whether `MPI.Finalize()` is called at the end of `train!`.
    Defaults to `true`. Set it to `false` when you want to run several `train!`
    calls (or other MPI work) in the same process.

```julia
# any optimizer, distributed the same way
Options(OptimMode; algorithm = Optim.ParticleSwarm(), parallel = MPIBackend())
Options(NLoptMode; algorithm = :LN_BOBYQA, parallel = MPIBackend())
Options(GradientMode; rule = Flux.Adam(0.1), parallel = MPIBackend(finalize = false))
```

### Return value

Only the controller process returns a meaningful [`Solution`](@ref
ApplicationDrivenLearning.Solution); worker processes return a placeholder and
their forecast model is left in an unspecified state. Guard any post-processing
accordingly.

!!! warning
    `finalize` defaults to `true`, and once MPI is finalized no MPI routine
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
        ApplicationDrivenLearning.GradientMode;
        epochs = 30,
        parallel = ApplicationDrivenLearning.MPIBackend(finalize = false),
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    if JQM.is_controller_process()
        println(sol.cost)
    end
    JQM.mpi_finalize()
    ```

The second form is also what you need when running several `train!` calls in
the same process: every call but the last must pass `finalize = false`.

!!! compat "Deprecated: NelderMeadMPIMode and GradientMPIMode"
    These were the only way to run over MPI before parallelism became an option,
    and only Nelder-Mead had one. They still work, forwarding to
    `parallel = MPIBackend()` and translating `mpi_finalize` onto the backend:

    ```julia
    # before                                    # now
    Options(NelderMeadMPIMode; iterations = 100)
    Options(OptimMode; algorithm = Optim.NelderMead(), iterations = 100,
            parallel = MPIBackend())

    Options(GradientMPIMode; epochs = 30)
    Options(GradientMode; epochs = 30, parallel = MPIBackend())
    ```

---

## Running over Distributed

Uses Julia's own `Distributed` stdlib, so there is nothing to install and no
external launcher. Unlike MPI, only the driver process runs `train!`; the workers
never execute your script.

That is the one difference that shows up in the API. The workers still need a
model of their own to solve against, and since they never ran your script they
have no way to obtain one — so instead of handing `train!` a model, you hand it a
**function that builds one**:

```julia
using Distributed
addprocs(4)
@everywhere using ApplicationDrivenLearning, Flux, JuMP, HiGHS

# `@everywhere` so that the workers can call it too
@everywhere function build_case()
    m = ApplicationDrivenLearning.Model()
    # ... variables, constraints and objectives ...
    set_optimizer(m, HiGHS.Optimizer)
    ApplicationDrivenLearning.set_forecast_model(
        m,
        ApplicationDrivenLearning.PredictiveModel(
            Chain(Dense(1 => 1));
            output_names = [:demand],
        ),
    )
    return m
end

opt = ApplicationDrivenLearning.Options(
    ApplicationDrivenLearning.GradientMode,
    rule = Flux.Adam(0.1),
    epochs = 30,
    parallel = ApplicationDrivenLearning.DistributedBackend(),
)
sol = ApplicationDrivenLearning.train!(build_case, X, Y, opt)
```

`train!` calls it once for the driver's own model and each worker calls it once
for its own — so a single function describes the whole run, and the driver and the
workers cannot end up solving different problems. Passing a `Model` under this
backend is an error, since there would be no way to give the workers one.

The builder returns a **ready-to-solve** model: variables, constraints,
objectives, `set_optimizer` and `set_forecast_model`. It is called once per worker
per `train!`, never per sample, and must be visible on the workers
(`@everywhere`, or defined in a package they load).

The workers' *initial* predictive-model weights do not matter, so a random `init`
is fine: every evaluation applies parameters sent from the driver before solving.

This is the same function an MPI script already needs in order to have every rank
build an identical case, so an existing MPI setup can usually pass its builder
unchanged.

!!! warning
    The builder must *construct* the model rather than close over one. An
    `ApplicationDrivenLearning.Model` that has had `set_optimizer` called on it
    cannot be sent to a worker at all: the solver's internal pointers travel as
    raw addresses and the attempt is a `ReadOnlyMemoryError`, which takes the
    process down rather than raising a catchable error. Building inside the
    builder avoids this entirely.

### Getting the trained model back

The builder form leaves you without a reference to the model that was trained. The
fitted parameters are in the returned `Solution`, so a fitted model is one call
away:

```julia
sol = ApplicationDrivenLearning.train!(build_case, X, Y, opt)

model = build_case()
ApplicationDrivenLearning.apply_params(model.forecast, sol.params)
```

### Arguments

  - `workers`: which worker ids to use. Defaults to `Distributed.workers()`, read
    when training starts — so calling `addprocs` after building the `Options`
    still works. If there are no worker processes, training runs on the driver at
    serial speed and warns, since the usual cause is a forgotten `addprocs`.
  - `verify`: whether to check at start-up that the workers' models match the
    driver's. Defaults to `true`; see below.

```julia
DistributedBackend()
DistributedBackend(; workers = [2, 3])
DistributedBackend(; verify = false)
```

### The workers' models are checked against the driver's

Since one function builds them all, the only way they can disagree is if that
function is not **deterministic** — if it closes over mutable global state, or
branches on something that differs per process. Narrow, but silent: the run would
simply optimize the wrong problem. So `DistributedBackend` checks, once per
`train!`:

  - the models' sizes always: input and output size, the number of policy and
    forecast variables, and the number of predictive-model parameters;
  - the assessed cost of one sample at the starting parameters, when `verify` is
    `true`.

The cost comparison is the one that catches a difference the sizes cannot see,
which is why it is on by default. It is a smoke test rather than a proof — one
sample cannot reveal a difference that only shows on another — and it costs one
extra pair of solves on the driver. Pass `verify = false` to skip it.

If the driver's model has no optimizer attached, the cost comparison cannot run;
the sizes are still checked and a warning says the rest was skipped.

### Return value

Only the driver runs `train!`, so there is no rank-dependent behaviour and no
placeholder to guard against: `sol` is the real [`Solution`](@ref
ApplicationDrivenLearning.Solution).

The workers release their models when `train!` returns, including when it throws,
so a later `train!` with a different model never inherits a stale one.

### Cost of the first run

The first distributed evaluation makes each worker compile the JuMP, DiffOpt and
solver code paths for itself, which takes appreciably longer than the training
that follows. This is per worker *process*, not per `train!`, so it is paid once per
session and is worth keeping in mind when timing a short run.
