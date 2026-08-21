using MPI
import JobQueueMPI as JQM

"""
    MPIBackend(; finalize = true) <: AbstractParallelBackend

Distribute the per-sample cost and gradient evaluations across MPI processes with
JobQueueMPI.jl.

Every rank runs the same script and so builds its own copy of the model; the
controller then drives the optimizer while the workers serve evaluations. Only the
controller returns a meaningful [`Solution`](@ref).

`finalize` controls whether `MPI.Finalize()` is called when training ends. Set it
to `false` when several `train!` calls share one MPI session.
"""
struct MPIBackend <: AbstractParallelBackend
    finalize::Bool
end

MPIBackend(; finalize::Bool = true) = MPIBackend(finalize)

function _with_evaluator(
    body,
    parallel::MPIBackend,
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
)
    JQM.mpi_init()
    step(v) = _sample_step(model, X, Y, v[1], v[2], v[3])
    result = Solution(0.0, Float64[])

    if JQM.is_controller_process()
        function evaluate(θ, batch::SampleBatch; with_gradients::Bool = false)
            # the controller never executes jobs - `JobQueueMPI` has it dispatch
            # and collect only - so its own model would otherwise stay at the
            # previous θ, which the flat parameter gradient reads
            apply_params(model.forecast, θ)
            indices = batch.indices
            # one broadcast per `pmap`, matching the worker loop below. Each rank
            # holds the whole dataset, so only the indices travel
            MPI.bcast(false, MPI.COMM_WORLD)
            results = JQM.pmap(step, [[θ, i, with_gradients] for i in indices])
            C = sum(r[1] for r in results) / length(indices)
            if !with_gradients
                return C, nothing
            end
            # stack per-sample gradients into rows aligned with `indices`
            return C, reduce(vcat, [r[2]' for r in results])
        end

        # `try`/`finally` so the workers are released even when the body throws.
        # Without it a failed solve - which an optimizer exploring freely can
        # easily provoke - would leave every worker blocked in `bcast` forever,
        # turning a training failure into a hung job
        try
            result = body(evaluate)
        finally
            MPI.bcast(true, MPI.COMM_WORLD)
        end
    elseif JQM.is_worker_process()
        # serve evaluations until the controller says it is done
        while true
            if MPI.bcast(false, MPI.COMM_WORLD)
                break
            end
            JQM.pmap(step, [])
        end
    end

    JQM.mpi_barrier()
    if parallel.finalize
        JQM.mpi_finalize()
    end
    return result
end

"""
    _as_mpi_params(params)

Translate the parameters of a deprecated `*MPIMode` into the current spelling,
moving `mpi_finalize` onto an [`MPIBackend`](@ref).
"""
function _as_mpi_params(params::Dict{Symbol,Any})
    forwarded = copy(params)
    delete!(forwarded, :mpi_finalize)
    forwarded[:parallel] =
        MPIBackend(; finalize = get(params, :mpi_finalize, true))
    return forwarded
end
