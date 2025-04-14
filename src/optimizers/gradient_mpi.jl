using Flux
import ParametricOptInterface as POI
using MPI
import JobQueueMPI as JQM

function train_with_gradient_mpi!(
    model::Model,
    X::Matrix{<:Real},
    Y::Matrix{<:Real},
    params::Dict{Symbol,Any},
)
    # extract params
    rule = get(params, :rule, Flux.Descent())
    epochs = get(params, :epochs, 100)
    batch_size = get(params, :batch_size, -1)
    verbose = get(params, :verbose, true)
    compute_cost_every = get(params, :compute_cost_every, 1)
    mpi_finalize = get(params, :mpi_finalize, true)
    time_limit = get(params, :time_limit, Inf)

    JQM.mpi_init()

    # init parameters
    start_time = time()
    is_done = false
    best_C = Inf
    best_θ = []
    curr_C = 0.0
    trace = Array{Float64}(undef, epochs)
    dCdz = Vector{Float32}(undef, size(model.policy_vars, 1))
    dCdy = Vector{Float32}(undef, model.forecast.output_size)
    T = size(X)[1]
    stochastic = batch_size > 0
    compute_full_cost = true
    opt_state = Flux.setup(rule, model.forecast)

    # precompute batches
    batches = repeat(1:T, outer = (1, epochs))'
    if stochastic
        batches = rand(1:T, (epochs, batch_size))
    end

    # cost and gradient compute function
    function compute_cost_and_gradients(θ, i, compute_gradient::Bool)
        apply_params(model.forecast, θ)
        yhat = model.forecast(X[i, :])
        step_cost = compute_single_step_cost(model, Y[i, :], yhat)
        if compute_gradient
            step_grad = compute_single_step_gradient(model, dCdz, dCdy)
        else
            step_grad = nothing
        end

        return step_cost, step_grad
    end

    # call optim as the controller
    if JQM.is_controller_process()

        # main loop
        for epoch = 1:epochs
            compute_full_cost = epoch % compute_cost_every == 0

            # broadcast `is_done = false`
            MPI.bcast(is_done, MPI.COMM_WORLD)

            # extract current parameters
            curr_θ = extract_params(model.forecast)

            if stochastic
                epochx = X[batches[epoch, :], :]
                # compute stochastic gradient
                pmap_result_with_gradients = JQM.pmap(
                    (v) -> compute_cost_and_gradients(v[1], v[2], true),
                    [[curr_θ, i] for i in batches[epoch, :]],
                )
                dCdy =
                    sum([r[2] for r in pmap_result_with_gradients]) ./ batch_size

                if compute_full_cost
                    # broadcast `is_done = false` again
                    MPI.bcast(is_done, MPI.COMM_WORLD)

                    # compute full cost
                    pmap_result_without_gradients = JQM.pmap(
                        (v) ->
                            compute_cost_and_gradients(v[1], v[2], false),
                        [[curr_θ, i] for i = 1:T],
                    )
                    curr_C =
                        sum([r[1] for r in pmap_result_without_gradients]) ./ T
                end

            else
                epochx = X
                # compute full cost and gradient
                pmap_result = JQM.pmap(
                    (v) -> compute_cost_and_gradients(v[1], v[2], true),
                    [[curr_θ, i] for i = 1:T],
                )
                curr_C = sum([r[1] for r in pmap_result]) ./ T
                dCdy = sum([r[2] for r in pmap_result]) ./ T
            end

            if compute_full_cost
                # store and print cost
                trace[epoch] = curr_C
                if verbose
                    dtime = time() - start_time
                    println("Epoch $epoch | Time = $(round(dtime, digits=1))s | Cost = $(round(curr_C, digits=2))")
                end

                # evaluate if best model
                if curr_C <= best_C
                    best_C = curr_C
                    best_θ = curr_θ
                end
            end

            # check time limit reach
            if time() - start_time > time_limit
                break
            end

            # take gradient step (if not last epoch)
            apply_gradient!(model.forecast, dCdy, epochx, opt_state)
        end

        # release workers
        is_done = true
        MPI.bcast(is_done, MPI.COMM_WORLD)

        # fix best model
        apply_params(model.forecast, best_θ)

    elseif JQM.is_worker_process()
        # continuoslly call pmap until controller is done
        while true
            is_done = MPI.bcast(is_done, MPI.COMM_WORLD)
            if is_done
                break
            end
            JQM.pmap((v) -> compute_cost_and_gradients(v[1], v[2], true), [])
        end
    end

    JQM.mpi_barrier()
    if mpi_finalize
        JQM.mpi_finalize()
    end

    return Solution(best_C, best_θ)
end
