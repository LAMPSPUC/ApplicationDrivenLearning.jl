"""
    AbstractParallelBackend

How the per-sample work of a training run is distributed.

Every setting of a backend lives on the backend object.
The one exception is
`mpi_finalize`, which predates this design and is reachable only through the
deprecated `*MPIMode` aliases.

[`BilevelMode`](@ref) is deliberately outside this: it builds a single MIP over
all samples and never evaluates a per-sample cost, so it has nothing to
distribute.

Implementations: [`SerialBackend`](@ref), [`MPIBackend`](@ref),
[`DistributedBackend`](@ref).
"""
abstract type AbstractParallelBackend end

"""
    SerialBackend <: AbstractParallelBackend

Evaluate everything in the current process. The default.
"""
struct SerialBackend <: AbstractParallelBackend end

"""
    SampleBatch(indices, X, Y)

The set of samples one call to `evaluate` covers, described once.

Holds both the global row `indices` and the corresponding slices of the data,
because the backends need different halves: the serial one works on the slices
directly, while a distributed one sends indices and lets each worker read its own
copy of the full data.

Both halves are only ever derived from the same `indices`, by [`_batch`](@ref) or
[`_full_batch`](@ref), and that is the point of the type. Passed as two separate
arguments — as they were before — each backend silently ignored the half it did not
use, so an inconsistent pair would have produced a different answer depending on
which backend was selected, with nothing to catch it.

The inner check compares lengths only. It cannot verify that `X` really is
`X[indices, :]`; checking the values would cost more than the evaluation it
protects. What it does catch is both ways of getting this wrong by accident:
passing the unsliced data alongside a batch, or slicing with the wrong indices.
"""
struct SampleBatch{
    TI<:AbstractVector{Int},
    TX<:AbstractMatrix{<:Real},
    TY<:AbstractMatrix{<:Real},
}
    indices::TI
    X::TX
    Y::TY

    function SampleBatch(indices::TI, X::TX, Y::TY) where {TI,TX,TY}
        if size(X, 1) != length(indices) || size(Y, 1) != length(indices)
            throw(
                DimensionMismatch(
                    "A `SampleBatch` over $(length(indices)) sample(s) was given " *
                    "$(size(X, 1)) row(s) of `X` and $(size(Y, 1)) of `Y`; both " *
                    "must be the rows selected by `indices`. Build the batch with " *
                    "`_batch(X, Y, indices)` or `_full_batch(X, Y)` rather than " *
                    "assembling it directly.",
                ),
            )
        end
        return new{TI,TX,TY}(indices, X, Y)
    end
end

"""
    _batch(X, Y, indices)

The batch of samples `indices`, slicing the data to match.

`indices` may repeat a sample: stochastic batches are drawn with replacement, so
this is a list and never a set.
"""
_batch(X, Y, indices) = SampleBatch(indices, X[indices, :], Y[indices, :])

"""
    _full_batch(X, Y)

The batch covering every sample, without copying the data.
"""
_full_batch(X, Y) = SampleBatch(1:size(X, 1), X, Y)

"""
    _parallel_backend(params)

Backend requested through the `parallel` keyword, defaulting to serial.
"""
function _parallel_backend(params::Dict{Symbol,Any})
    parallel = get(params, :parallel, SerialBackend())
    if !(parallel isa AbstractParallelBackend)
        throw(
            ArgumentError(
                "`parallel` must be an `AbstractParallelBackend`, such as " *
                "`SerialBackend()` or `MPIBackend()`, got $(typeof(parallel)).",
            ),
        )
    end
    return parallel
end

"""
    _assert_model_is_enough(parallel)

Throw if `parallel` cannot work from a [`Model`](@ref) alone.

Only [`DistributedBackend`](@ref) can't: its workers never run the caller's
script, so they need a builder function to obtain a model of their own. Every
other backend is satisfied by the model it is given.
"""
_assert_model_is_enough(::AbstractParallelBackend) = nothing

"""
    _with_backend_builder(parallel, build_model)

`parallel` with `build_model` attached, for the backends that need it, and
`parallel` itself for those that do not.

The builder is supplied by the caller to `train!`, but it has to reach
`_with_evaluator`, which receives the backend rather than the options — so
`train!` puts it on the backend. That keeps `_train!`'s signature intact, which
matters because it is the extension point package extensions implement, and it
keeps every backend setting on the backend object.
"""
_with_backend_builder(parallel::AbstractParallelBackend, ::Any) = parallel

"""
    _with_model_builder(options, build_model)

`options` with the model builder attached to its backend, or `options` unchanged
when the backend has no use for one.
"""
function _with_model_builder(options::Options, build_model)
    parallel = _parallel_backend(options.params)
    updated = _with_backend_builder(parallel, build_model)
    updated === parallel && return options
    params = copy(options.params)
    params[:parallel] = updated
    return Options(options.mode; params...)
end

"""
    _sample_step(model, X, Y, θ, i, with_gradients)

Assessed cost of sample `i` at parameters `θ`, and its gradient with respect to
the forecasts when asked for.

This is the unit of work that gets distributed. It takes `model`, `X` and `Y` as
arguments rather than reading them from anywhere, which is what lets a worker
supply its own copies — under Distributed the driver's model cannot be sent at all.

`with_gradients` travels with the job rather than being baked into the closure:
under JobQueueMPI it is the *worker's* closure that executes, so a flag fixed on
the worker side would have it computing gradients the controller never asked for —
which is what the previous implementation did on every full-cost pass, paying a
DiffOpt reverse pass per sample for a result that was discarded.
"""
function _sample_step(model::Model, X, Y, θ, i::Int, with_gradients::Bool)
    apply_params(model.forecast, θ)
    yhat = model.forecast(X[i, :])
    cost = _compute_single_step_cost(model, Y[i, :], yhat)
    if !with_gradients
        return cost, nothing
    end
    dCdz = Vector{Float64}(undef, length(model.policy_vars))
    dCdy = Vector{Float64}(undef, model.forecast.output_size)
    # `_compute_single_step_gradient` returns the shared `dCdy` buffer, so it has
    # to be copied before being handed back to the caller
    return cost, copy(_compute_single_step_gradient(model, dCdz, dCdy))
end

"""
    _with_evaluator(body, parallel, model, X, Y)

Run `body(evaluate)` under `parallel`, where

    evaluate(θ, batch::SampleBatch; with_gradients = false) -> (C, dC)

gives the mean assessed cost over `batch` and, when asked, the
`(length(batch.indices), output_size)` matrix of per-sample gradients with respect
to the forecasts, its rows aligned with `batch.indices`.

Whatever `body` returns is the return value here. A scope rather than a plain
function because a backend may need to wrap the whole run: MPI has workers serving
evaluations until the controller releases them, and Distributed builds worker-side
models on entry and drops them on exit, so setup, teardown and any
controller/worker split all have to live around `body`.

Backends differ in *who* runs `body`, and that is a property of each rather than of
this seam. Serial and Distributed run it in the calling process. Under
[`MPIBackend`](@ref) every rank calls `train!`, `body` runs on the controller only,
and the other ranks get a placeholder [`Solution`](@ref) back.

Whichever backend is in use, `body` may assume that `evaluate(θ, …)` leaves
`model.forecast` at `θ` in *this* process — the gradient-based trainers form
`dC/dθ` here from the per-sample `dC/dŷ` the workers return, and that reads these
parameters.
"""
function _with_evaluator(
    body,
    ::SerialBackend,
    model::Model,
    # the full data is part of the seam - the distributing backends ship it to their
    # workers up front - but serial reads only what each batch carries
    ::AbstractMatrix{<:Real},
    ::AbstractMatrix{<:Real},
)
    function evaluate(θ, batch::SampleBatch; with_gradients::Bool = false)
        apply_params(model.forecast, θ)
        if with_gradients
            return _compute_cost_on_matrices(model, batch.X, batch.Y, true)
        end
        return _compute_cost_on_matrices(model, batch.X, batch.Y, false),
        nothing
    end
    return body(evaluate)
end
