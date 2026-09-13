using Flux

"""
    _train!(::Type{GradientMode}, model, X, Y, params)

Train the predictive model with first-order updates driven by the gradient of
the assessed cost with respect to the forecasts.

Runs for at most `epochs` iterations, keeping the parameters with the lowest
cost seen, and stops early on `time_limit` or `g_tol`. See [`GradientMode`](@ref)
for the accepted `params`.

The loop is backend-independent: only the cost and per-sample gradient evaluation
is distributed, through [`_with_evaluator`](@ref), so the same body serves the
serial and the MPI runs. Under MPI it executes on the controller only.
"""
function _train!(
    ::Type{GradientMode},
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
    parallel = _parallel_backend(params)

    T = size(X, 1)
    stochastic = batch_size > 0

    # precompute batches (only needed in the stochastic case; the deterministic
    # branch uses the full dataset directly). NOTE: this must stay a single
    # `rand` call, made before the backend splits ranks, so that a seeded serial
    # run and a seeded MPI run draw the same batches - `test/mpi/mpi_modes.jl`
    # compares their parameters directly
    batches = stochastic ? rand(1:T, (epochs, batch_size)) : zeros(Int, 0, 0)

    return _with_evaluator(parallel, model, X, Y) do evaluate
        start_time = time()
        best_C = Inf
        best_θ = extract_params(model.forecast)
        opt_state = Flux.setup(rule, model.forecast)
        # built once: the whole dataset does not change between epochs, and it is
        # a view onto `X`/`Y` rather than a copy
        full = _full_batch(X, Y)

        # main loop
        for epoch = 1:epochs
            compute_full_cost = epoch % compute_cost_every == 0
            curr_θ = extract_params(model.forecast)

            if stochastic
                batch = _batch(X, Y, view(batches, epoch, :))
                epochx = batch.X
                C, dC = evaluate(curr_θ, batch; with_gradients = true)
                if compute_full_cost
                    # the gradient refers to the batch; the reported cost is
                    # recomputed over the whole dataset
                    C, _ = evaluate(curr_θ, full)
                end
            else
                epochx = X
                C, dC = evaluate(curr_θ, full; with_gradients = true)
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
                    best_θ = curr_θ
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
end

function _train!(
    ::Type{GradientMPIMode},
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::Dict{Symbol,Any},
)
    return _train!(GradientMode, model, X, Y, _as_mpi_params(params))
end

"""
    _train_with_gradient!(model, X, Y, params)

Deprecated. Retained because it was reachable as
`ApplicationDrivenLearning._train_with_gradient!`; use `train!` with
[`GradientMode`](@ref).
"""
function _train_with_gradient!(
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::Dict{Symbol,Any},
)
    return _train!(GradientMode, model, X, Y, params)
end
