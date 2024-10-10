using Flux
import ParametricOptInterface as POI
using MPI
import JobQueueMPI as JQM


function train_with_gradient_mpi!(
    model::Model,
    X::Matrix{<:Real},
    Y::Matrix{<:Real},
    params::Dict{Symbol, Any}
)
    # extract params
    rule = get(params, :rule, Flux.Descent())
    epochs = get(params, :epochs, 100)
    batch_size = get(params, :batch_size, -1)

    JQM.mpi_init()
    
    # init parameters
    is_done = false
    best_C = Inf
    best_θ = []
    curr_C = 0.0
    trace = Array{Float64}(undef, epochs)
    dCdz = Vector{Float32}(undef, size(model.policy_vars, 1))
    dCdy = Vector{Float32}(undef, model.forecast.output_size)
    T = size(X)[1]

    # cost and gradient compute function
    function compute_cost_and_gradients(θ, i)
        apply_params(model.forecast, θ)
        yhat = model.forecast(X[i, :])
        step_cost = compute_single_step_cost(model, Y[i, :], yhat)
        step_grad = compute_single_step_gradient(model, dCdz, dCdy)
              
        return step_cost, step_grad
        
    end

    # call optim as the controller
    if JQM.is_controller_process()

        # main loop
        for epoch=1:epochs
            println("Epoch $epoch")
            MPI.bcast(is_done, MPI.COMM_WORLD)

            # define batch
            if batch_size > 0
                batch = rand(1:T, batch_size)
            else
                batch = 1:T
            end
            
            # compute cost and gradient
            curr_θ = extract_params(model.forecast)

            pmap_result = JQM.pmap(
                (v) -> compute_cost_and_gradients(v[1], v[2]), 
                [[curr_θ, i] for i in batch]
            )
            curr_C = sum([r[1] for r in pmap_result])
            dCdy = sum([r[2] for r in pmap_result])
            trace[epoch] = curr_C
            println("Cost: $curr_C")

            # evaluate if best model
            if curr_C <= best_C
                best_C = curr_C
                best_θ = curr_θ
            end
    
            # take gradient step (if not last epoch)
            if epoch < epochs
                apply_gradient!(model.forecast, dCdy, rule)
            end
        end
        
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
            JQM.pmap((v) -> compute_cost_and_gradients(v[1], v[2]), [])
        end
    end

    JQM.mpi_barrier()
    JQM.mpi_finalize()

    return Solution(best_C, best_θ)
end
