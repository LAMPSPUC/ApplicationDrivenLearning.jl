using Flux

"""
    _stochastic_compute(model, X, Y, epochx, batch, compute_full_cost)

Compute the assess cost and the cost gradient (with respect to the predicted
values) on a subset `batch` of the examples. `epochx` must be `X[batch, :]`,
which the caller already needs for the gradient step and so passes in rather
than having it built twice.

When `compute_full_cost` is `true`, the returned cost is recomputed over the
whole dataset — the gradient still refers to the batch only.
"""
function _stochastic_compute(
    model,
    X,
    Y,
    epochx,
    batch,
    compute_full_cost::Bool,
)
    C, dC = compute_cost(model, epochx, Y[batch, :], true)
    if compute_full_cost
        C = compute_cost(model, X, Y, false)
    end
    return C, dC
end

"""
    _deterministic_compute(model, X, Y)

Compute the assess cost and the cost gradient (with respect to the predicted
values) on the complete set of examples.
"""
function _deterministic_compute(model, X, Y)
    C, dC = compute_cost(model, X, Y, true)
    return C, dC
end

"""
    _train_with_gradient!(model, X, Y, params)

Train the predictive model with first-order updates driven by the gradient of
the assessed cost with respect to the forecasts.

Runs for at most `epochs` iterations, keeping the parameters with the lowest
cost seen, and stops early on `time_limit` or `g_tol`. See [`GradientMode`](@ref)
for the accepted `params`.
"""
function _train_with_gradient!(
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::Dict{Symbol,Any},
)
    # extract params
    rule = get(params, :rule, Flux.Descent())
    epochs = get(params, :epochs, 100)
    batch_size = get(params, :batch_size, -1)
    verbose = get(params, :verbose, true)
    compute_cost_every = get(params, :compute_cost_every, 1)
    time_limit = get(params, :time_limit, Inf)
    g_tol = get(params, :g_tol, 0)

    # init parameters
    start_time = time()
    T = size(X)[1]
    best_C = Inf
    best_θ = extract_params(model.forecast)
    stochastic = batch_size > 0
    opt_state = Flux.setup(rule, model.forecast)

    # precompute batches (only needed in the stochastic case; the
    # deterministic branch uses the full dataset directly)
    batches = stochastic ? rand(1:T, (epochs, batch_size)) : zeros(Int, 0, 0)

    # main loop
    for epoch = 1:epochs
        compute_full_cost = epoch % compute_cost_every == 0

        if stochastic
            batch = view(batches, epoch, :)
            epochx = X[batch, :]
            C, dC = _stochastic_compute(
                model,
                X,
                Y,
                epochx,
                batch,
                compute_full_cost,
            )
        else
            epochx = X
            C, dC = _deterministic_compute(model, X, Y)
        end

        if compute_full_cost
            # print cost
            if verbose
                dtime = time() - start_time
                println(
                    "Epoch $epoch | Time = $(round(dtime, digits=1))s | Cost = $(round(C, digits=2))",
                )
            end

            # evaluate if best model
            if C <= best_C
                best_C = C
                best_θ = extract_params(model.forecast)
            end
        end

        # check time limit reach
        if time() - start_time > time_limit
            if verbose
                println("Time limit reached.")
            end
            break
        end

        # check gradient tolerance
        if maximum(abs, dC) < g_tol
            if verbose
                println("Gradient tolerance reached.")
            end
            break
        end

        # take gradient step
        apply_gradient!(model.forecast, dC, epochx, opt_state)
    end

    # fix best model
    apply_params(model.forecast, best_θ)

    return Solution(best_C, best_θ)
end

"""
    _train!(::Type{GradientMode}, model, X, Y, params)

Dispatch entry point for [`GradientMode`](@ref); see
[`_train_with_gradient!`](@ref).
"""
function _train!(
    ::Type{GradientMode},
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::Dict{Symbol,Any},
)
    return _train_with_gradient!(model, X, Y, params)
end
