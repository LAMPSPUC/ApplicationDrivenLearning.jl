using Optim
using MPI
import JobQueueMPI as JQM

function train_with_nelder_mead_mpi!(
    model::Model,
    X::Matrix{<:Real},
    Y::Matrix{<:Real},
    params::Dict{Symbol,Any},
)
    JQM.mpi_init()

    # extract params
    mpi_finalize = get(params, :mpi_finalize, true)
    delete!(params, :mpi_finalize)
    optim_options = Optim.Options(; params...)

    is_done = false
    res = nothing
    final_sol = []
    final_cost = 0.0
    T = size(X)[1]
    function compute_cost(θ, i)
        apply_params(model.forecast, θ)
        yhat = model.forecast(X[i, :])
        return compute_single_step_cost(model, Y[i, :], yhat)
    end

    # call optim as the controller
    if JQM.is_controller_process()

        # run optimization
        initial_sol = extract_params(model.forecast)
        res = Optim.optimize(initial_sol, NelderMead(), optim_options) do θ
            MPI.bcast(is_done, MPI.COMM_WORLD)
            c_θ = JQM.pmap(
                (v) -> compute_cost(v[1], v[2]),
                [[θ, i] for i = 1:T],
            )
            return sum(c_θ) ./ T
        end

        # print solution
        println("Final solution: $(Optim.minimizer(res))")

        # update model parameters
        final_sol = Optim.minimizer(res)
        apply_params(model.forecast, final_sol)

        # get cost
        final_cost = minimum(res)

        # release workers
        is_done = true
        MPI.bcast(is_done, MPI.COMM_WORLD)

    elseif JQM.is_worker_process()
        # continuoslly call pmap until controller is done
        while true
            is_done = MPI.bcast(is_done, MPI.COMM_WORLD)
            if is_done
                break
            end
            JQM.pmap((v) -> compute_cost(v[1], v[2]), [])
        end
    end

    JQM.mpi_barrier()
    if mpi_finalize
        JQM.mpi_finalize()
    end

    return Solution(final_cost, final_sol)
end
