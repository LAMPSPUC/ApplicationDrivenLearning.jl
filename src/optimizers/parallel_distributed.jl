using Distributed

"""
    DistributedBackend(; workers = nothing, verify = true) <: AbstractParallelBackend

Distribute the per-sample cost and gradient evaluations across `Distributed`
worker processes.

Unlike [`MPIBackend`](@ref), where every rank runs the same script and so builds
its own model, here only the driver runs `train!` — so the workers need a way to
obtain a model of their own. That is why this backend must be used with the
builder form of [`train!`](@ref):

```julia
sol = train!(
    build_case,
    X,
    Y,
    Options(GradientMode; parallel = DistributedBackend()),
)
```

Passing a `Model` instead is an error. The builder is called once per worker per
`train!`, never per sample, and must be defined on the workers (`@everywhere`, or
in a package they load).

It must also *construct* the model rather than close over one: an
`ApplicationDrivenLearning.Model` that has had `set_optimizer` called on it cannot
be serialized at all — the attempt is a `ReadOnlyMemoryError` that takes the
process down, since the solver's C pointers travel as raw addresses.

# Keyword arguments

  - `workers`: the worker ids to use. Defaults to `Distributed.workers()` read when
    training starts, so calling `addprocs` after building the [`Options`](@ref)
    still works.

  - `verify`: check at start-up that the workers' models match the driver's. Their
    sizes are always compared, which is free; `verify` additionally compares the
    assessed cost of one sample, which costs one extra pair of solves on the
    driver.

    Since the driver and the workers now build from the *same* function, this
    guards a narrower thing than it once did: a builder that is not
    deterministic — one closing over mutable global state, or whose structure
    varies per process. Cheap enough to leave on.

# Example

```julia
using Distributed
addprocs(4)
@everywhere using ApplicationDrivenLearning, HiGHS, Flux
@everywhere function build_case()
    m = ApplicationDrivenLearning.Model()
    @variable(m, demand, ApplicationDrivenLearning.Forecast)
    # ... policy variables, constraints, objectives ...
    set_optimizer(m, HiGHS.Optimizer)
    ApplicationDrivenLearning.set_forecast_model(
        m,
        ApplicationDrivenLearning.ForecastModel(
            architecture = Dense(1 => 1),
            outputs = [demand],
        ),
    )
    return m
end

sol = train!(
    build_case,
    X,
    Y,
    Options(GradientMode; parallel = DistributedBackend()),
)
```
"""
struct DistributedBackend <: AbstractParallelBackend
    # filled in by `train!` from the builder it was given, not by the user
    builder::Any
    workers::Union{Vector{Int},Nothing}
    verify::Bool
end

function DistributedBackend(;
    workers::Union{Vector{Int},Nothing} = nothing,
    verify::Bool = true,
)
    return DistributedBackend(nothing, workers, verify)
end

function _assert_model_is_enough(parallel::DistributedBackend)
    if isnothing(parallel.builder)
        throw(
            ArgumentError(
                "`DistributedBackend` needs a model builder, not a model: its " *
                "workers never run your script, and a model that has had " *
                "`set_optimizer` called on it cannot be sent to them. Call " *
                "`train!(build_model, X, Y, options)` with a zero-argument " *
                "function returning a ready-to-solve model, defined on the " *
                "workers with `@everywhere`.",
            ),
        )
    end
    return nothing
end

function _with_backend_builder(parallel::DistributedBackend, build_model)
    return DistributedBackend(build_model, parallel.workers, parallel.verify)
end

"""
    ApplicationDrivenLearning._WORKER_STATE

The model and data a `Distributed` worker evaluates samples against, or `nothing`
when it holds none.

Set once per `train!` by [`_worker_materialize`](@ref) and cleared by
[`_worker_release`](@ref), so a later `train!` with a different model cannot
inherit a stale one. One slot per process, which means two `train!` calls sharing a
worker pool *concurrently* would clash; the driver is single-threaded, so that
requires deliberately threading it.
"""
const _WORKER_STATE = Ref{Any}(nothing)

"""
    _model_shape(model)

The handful of sizes that must agree between a worker's model and the driver's.

Includes the parameter count because of an asymmetry in
[`apply_params`](@ref): it walks the model's *own* parameters and indexes into `θ`,
so a `θ` longer than the network needs is silently truncated while a shorter one
throws. A worker network smaller than the driver's is therefore the direction that
fails quietly, and it is exactly what a builder using a different layer width
produces.
"""
function _model_shape(model::Model)
    return (
        input_size = model.forecast.input_size,
        output_size = model.forecast.output_size,
        n_policy = length(model.policy_vars),
        n_forecast = length(model.forecast_vars),
        n_params = length(extract_params(model.forecast)),
    )
end

"""
    _worker_materialize(builder, X, Y)

Build this worker's model and keep it, along with the data, for the rest of the
training run. Returns its [`_model_shape`](@ref) for the driver to check.

`X` and `Y` arrive once, here, rather than being sliced into every job: they do not
change during training.
"""
function _worker_materialize(builder, X, Y)
    model = builder()
    _assert_forecast_model_set(model)
    # under MPI this happens because every rank runs `train!`; there is no `train!`
    # on a Distributed worker, and `_compute_single_step_cost` requires it
    _build(model)
    _WORKER_STATE[] = (model, X, Y)
    return _model_shape(model)
end

"""
    _worker_release()

Drop this worker's model and data.
"""
function _worker_release()
    _WORKER_STATE[] = nothing
    return nothing
end

"""
    _worker_chunk(θ, indices, with_gradients)

Evaluate a contiguous run of samples on this worker, in the order given.
"""
function _worker_chunk(θ, indices, with_gradients::Bool)
    state = _WORKER_STATE[]
    if isnothing(state)
        error(
            "This worker holds no model. `_worker_materialize` must run before " *
            "any evaluation; this usually means the worker was restarted, or " *
            "released, mid-training.",
        )
    end
    # `_WORKER_STATE` is untyped, so hand off through a function barrier rather
    # than looping here: otherwise every `_sample_step` call is a dynamic dispatch
    return _chunk_steps(
        state[1]::Model,
        state[2],
        state[3],
        θ,
        indices,
        with_gradients,
    )
end

function _chunk_steps(model::Model, X, Y, θ, indices, with_gradients::Bool)
    return [_sample_step(model, X, Y, θ, i, with_gradients) for i in indices]
end

"""
    _worker_pool(parallel::DistributedBackend)

Worker ids to spread the samples over.
"""
function _worker_pool(parallel::DistributedBackend)
    pool =
        isnothing(parallel.workers) ? Distributed.workers() : parallel.workers
    if isempty(pool)
        # only reachable by passing `workers = Int[]` explicitly, since
        # `Distributed.workers()` reports the driver rather than nothing
        throw(
            ArgumentError(
                "`DistributedBackend` was given an empty list of workers, so " *
                "there is nowhere to evaluate samples.",
            ),
        )
    end
    if pool == [1]
        @warn "`DistributedBackend` found no worker processes, so every sample " *
              "will be evaluated on the driver at serial speed. Add workers " *
              "with `Distributed.addprocs` before training."
    end
    return pool
end

"""
    _split_indices(indices, n)

Split `indices` into at most `n` contiguous runs of near-equal length, preserving
order and repeats.

Order is what keeps the returned per-sample gradients aligned with the batch, and
repeats occur because stochastic batches are drawn with replacement — so this
treats `indices` as a list, never as a set. No run is empty: a batch smaller than
the pool simply uses fewer workers.
"""
function _split_indices(indices::AbstractVector{Int}, n::Int)
    total = length(indices)
    n = min(n, total)
    n < 1 && return Vector{Vector{Int}}()
    base, extra = divrem(total, n)
    chunks = Vector{Vector{Int}}(undef, n)
    lo = 1
    for k = 1:n
        len = base + (k <= extra ? 1 : 0)
        chunks[k] = collect(indices[lo:lo+len-1])
        lo += len
    end
    return chunks
end

"""
    _verify_worker_models(model, shapes, pool, verify, X, Y)

Check that the workers are solving the same problem as the driver.

Nothing else does. Under MPI a divergence needs rank-dependent code, but here a
builder that differs from the driver's model by one cost coefficient is a one-line
mistake, and its consequence is plausible, entirely wrong training with no error
ever raised.

Two checks. The shape comparison is free and always runs. The value comparison —
one sample's assessed cost at `θ₀`, driver against worker — is what actually
catches a drifted builder, and costs one plan solve plus one assess solve on the
driver. It is a smoke test rather than a proof: one sample cannot reveal a
difference that only shows on another, such as a constraint binding at high demand
only. It catches whole-problem mistakes, which are the ones people make.
"""
function _verify_worker_models(
    model::Model,
    shapes,
    pool,
    verify::Bool,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    driver = _model_shape(model)
    for (w, shape) in zip(pool, shapes)
        if shape != driver
            differing = [
                "$k: worker $(shape[k]) vs driver $(driver[k])" for
                k in keys(driver) if shape[k] != driver[k]
            ]
            throw(
                ArgumentError(
                    "The model built by the `DistributedBackend` builder on " *
                    "worker $w does not match the model being trained: " *
                    "$(join(differing, "; ")). The builder must construct the " *
                    "same problem, with the same predictive model, as the model " *
                    "passed to `train!`.",
                ),
            )
        end
    end

    verify || return nothing

    θ0 = extract_params(model.forecast)
    worker_cost =
        remotecall_fetch(_worker_chunk, first(pool), θ0, [1], false)[1][1]
    # the driver never solves otherwise under this backend, so a model without an
    # optimizer attached is possible and legitimate. Skip loudly rather than fail:
    # if the driver genuinely cannot solve at θ₀ the workers will say so on the
    # first real evaluation anyway
    driver_cost = try
        _sample_step(model, X, Y, θ0, 1, false)[1]
    catch err
        @warn "`DistributedBackend` could not evaluate the driver's own model, " *
              "so the builder was only checked for matching shapes. Pass " *
              "`verify = false` to skip this check deliberately." exception =
            err
        return nothing
    end

    if !isapprox(worker_cost, driver_cost; rtol = 1e-6)
        throw(
            ArgumentError(
                "The model built by the `DistributedBackend` builder gives a " *
                "different cost from the model being trained: worker " *
                "$worker_cost vs driver $driver_cost on sample 1 at the " *
                "starting parameters. The two describe different problems, so " *
                "training would optimize the wrong one.",
            ),
        )
    end
    return nothing
end

function _with_evaluator(
    body,
    parallel::DistributedBackend,
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    pool = _worker_pool(parallel)
    shapes = asyncmap(pool) do w
        return remotecall_fetch(_worker_materialize, w, parallel.builder, X, Y)
    end

    # `try`/`finally` for the same reason `MPIBackend` has one: an optimizer
    # exploring freely can provoke a failed solve, and the workers must not be left
    # holding a model - a later `train!` would otherwise find a stale one
    try
        _verify_worker_models(model, shapes, pool, parallel.verify, X, Y)

        function evaluate(θ, batch::SampleBatch; with_gradients::Bool = false)
            # the workers evaluate at θ, but `dC/dθ` is formed here from the
            # `dC/dŷ` they return, and that reads the driver's own parameters
            apply_params(model.forecast, θ)
            indices = batch.indices
            chunks = _split_indices(indices, length(pool))
            # one transfer of θ per worker per evaluation rather than one per
            # sample, which is what `pmap` over samples would cost. `asyncmap`
            # keeps the results in chunk order, which is what keeps the gradient
            # rows aligned with `indices`
            parts = asyncmap(zip(pool, chunks)) do (w, chunk)
                return remotecall_fetch(_worker_chunk, w, θ, chunk, with_gradients)
            end
            results = reduce(vcat, parts)
            C = sum(r[1] for r in results) / length(indices)
            if !with_gradients
                return C, nothing
            end
            return C, reduce(vcat, [r[2]' for r in results])
        end

        return body(evaluate)
    finally
        # a worker that has died makes releasing it throw, and a throw from here
        # would replace whatever actually went wrong with a message about cleanup
        try
            asyncmap(w -> remotecall_fetch(_worker_release, w), pool)
        catch err
            @warn "`DistributedBackend` could not release every worker; one may " *
                  "still be holding a model." exception = err
        end
    end
end
