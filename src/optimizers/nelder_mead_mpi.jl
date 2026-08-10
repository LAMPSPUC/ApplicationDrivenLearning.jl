using Optim
using MPI
import JobQueueMPI as JQM

"""
    _train_with_nelder_mead_mpi!(model, X, Y, params)

MPI counterpart of `_train_with_nelder_mead!`: the controller process
runs Nelder-Mead while the per-sample cost evaluations are distributed over
the worker processes with JobQueueMPI.jl.

Only the controller returns a meaningful [`Solution`](@ref). See
[`NelderMeadMPIMode`](@ref) for the accepted `params`.
"""
function _train_with_nelder_mead_mpi!(
    model::Model,
    X::Matrix{<:Real},
    Y::Matrix{<:Real},
    params::Dict{Symbol,Any},
)
    JQM.mpi_init()

    # extract params
    mpi_finalize = get(params, :mpi_finalize, true)
    optim_params = filter(x -> x[1] != :mpi_finalize, params)
    optim_options = Optim.Options(; optim_params...)

    is_done = false
    res = nothing
    final_sol = Float64[]
    final_cost = 0.0
    T = size(X)[1]
    function step_cost(θ, i)
        apply_params(model.forecast, θ)
        yhat = model.forecast(X[i, :])
        return _compute_single_step_cost(model, Y[i, :], yhat)
    end

    # call optim as the controller
    if JQM.is_controller_process()

        # run optimization
        initial_sol = extract_params(model.forecast)
        res = Optim.optimize(initial_sol, NelderMead(), optim_options) do θ
            MPI.bcast(is_done, MPI.COMM_WORLD)
            c_θ =
                JQM.pmap((v) -> step_cost(v[1], v[2]), [[θ, i] for i = 1:T])
            return sum(c_θ) ./ T
        end

        # print solution, following the same flag that drives Optim's trace
        if get(params, :show_trace, false)
            println("Final solution: $(Optim.minimizer(res))")
        end

        # update model parameters
        final_sol = Optim.minimizer(res)
        apply_params(model.forecast, final_sol)

        # get cost
        final_cost = Optim.minimum(res)

        # release workers
        is_done = true
        MPI.bcast(is_done, MPI.COMM_WORLD)

    elseif JQM.is_worker_process()
        # continuously call pmap until controller is done
        while true
            is_done = MPI.bcast(is_done, MPI.COMM_WORLD)
            if is_done
                break
            end
            JQM.pmap((v) -> step_cost(v[1], v[2]), [])
        end
    end

    JQM.mpi_barrier()
    if mpi_finalize
        JQM.mpi_finalize()
    end

    return Solution(final_cost, final_sol)
end
