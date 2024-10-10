using Flux

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

    # init parameters
    T = size(X)[1]
    best_C = Inf
    best_θ = extract_params(model.forecast)
    trace = Array{Float64}(undef, epochs)

    # main loop
    for epoch=1:epochs

        # define batch
        if batch_size > 0
            batch = rand(1:T, batch_size)
        else
            batch = 1:T
        end

        # compute cost and gradients
        C, dC = compute_cost(model, X[batch, :], Y[batch, :], true)
        trace[epoch] = C

        # evaluate if best model
        if C <= best_C
            best_C = C
            best_θ = extract_params(model.forecast)
        end

        # take gradient step
        apply_gradient!(model.forecast, dC, rule)

        # print epoch cost
        if verbose
            println("Epoch $epoch | Cost = $(round(C, digits=2))")
        end
    end

    # fix best model
    apply_params(model.forecast, best_θ)

    return Solution(best_C, best_θ)
end
