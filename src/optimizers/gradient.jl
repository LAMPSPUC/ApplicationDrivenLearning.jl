using Flux

"""
Compute assess cost and cost gradient (with respect to predicted values) based on incomplete batch of examples.
"""
function stochastic_compute(model, X, Y, batch, compute_full_cost::Bool)
    C, dC = compute_cost(model, X[batch, :], Y[batch, :], true)
    if compute_full_cost
        C = compute_cost(model, X, Y, false)
    end
    return C, dC
end

"""
Compute assess cost and cost gradient (with respect to predicted values) based on complete batch of examples.
"""
function deterministic_compute(model, X, Y)
    C, dC = compute_cost(model, X, Y, true)
    return C, dC
end

function train_with_gradient!(  
    model::Model,
    X::Matrix{<:Real},
    Y::Matrix{<:Real},
    params::Dict{Symbol, Any}
)
    # extract params
    rule = get(params, :rule, Flux.Descent())
    epochs = get(params, :epochs, 100)
    batch_size = get(params, :batch_size, -1)
    verbose = get(params, :verbose, true)
    compute_cost_every = get(params, :compute_cost_every, 1)

    # init parameters
    T = size(X)[1]
    best_C = Inf
    best_θ = extract_params(model.forecast)
    trace = Array{Float64}(undef, epochs)
    stochastic = batch_size > 0
    
    # precompute batches
    batches = repeat(1:T, outer=(1, epochs))'
    if stochastic
        batches = rand(1:T, (epochs, batch_size))
    end

    # main loop
    for epoch=1:epochs
        compute_full_cost = epoch % compute_cost_every == 0

        if stochastic
            epochx = X[batches[epoch, :], :]
            C, dC = stochastic_compute(model, X, Y, batches[epoch, :], compute_full_cost)
        else
            epochx = X
            C, dC = deterministic_compute(model, X, Y)
        end
        
        if compute_full_cost
            # store and print cost
            trace[epoch] = C
            if verbose
                println("Epoch $epoch | Cost = $(round(C, digits=2))")
            end

            # evaluate if best model
            if C <= best_C
                best_C = C
                best_θ = extract_params(model.forecast)
            end
        end

        # take gradient step
        apply_gradient!(model.forecast, dC, epochx, rule)

    end

    # fix best model
    apply_params(model.forecast, best_θ)

    return Solution(best_C, best_θ)
end
