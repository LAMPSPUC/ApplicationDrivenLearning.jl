using Optim

"""
    _optim_algorithm(params)

Algorithm object to hand to `Optim.optimize`, defaulting to Nelder-Mead.

Kept separate so that [`NelderMeadMode`](@ref) can build its algorithm from the
`initial_simplex` and `parameters` keywords it used to accept directly, while
[`OptimMode`](@ref) takes the algorithm object whole.
"""
function _optim_algorithm(params::Dict{Symbol,Any})
    return get(params, :algorithm, Optim.NelderMead())
end

# keywords consumed here rather than forwarded to `Optim.Options`
const _OPTIM_RESERVED = (
    :algorithm,
    :initial_simplex,
    :parameters,
    :lower_bounds,
    :upper_bounds,
    :parallel,
    :mpi_finalize,
)

"""
    _train_with_optim!(model, X, Y, params, algorithm)

Train the predictive model with an Optim.jl `algorithm`, using the assessed cost
as the objective.

The objective is the flat parameter vector in, scalar cost out — exactly the
interface `Optim.optimize` expects, so nothing here is specific to a particular
algorithm. Derivative-free algorithms use only `fitness`; gradient-based ones
(`LBFGS`, `BFGS`, `ConjugateGradient`, …) additionally get `g!`, which is what
makes them reachable at all.

`g!` takes its per-sample `dC/dŷ` from `evaluate` and turns it into `dC/dθ`
with `_flat_parameter_gradient`. It must get the cost from `evaluate` rather
than calling `compute_cost` directly: `evaluate` is the seam the parallel
backends replace, so only that route distributes.

See [`OptimMode`](@ref) for the accepted `params`; any key other than those in
`_OPTIM_RESERVED` is forwarded to `Optim.Options`.
"""
function _train_with_optim!(
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::Dict{Symbol,Any},
    algorithm,
)
    optim_params = filter(x -> !(x[1] in _OPTIM_RESERVED), params)
    optim_options = Optim.Options(; optim_params...)
    parallel = _parallel_backend(params)

    return _with_evaluator(parallel, model, X, Y) do evaluate
        # every evaluation here covers the whole dataset
        full = _full_batch(X, Y)

        # fitness function
        function fitness(θ)
            return _evaluate_or_explain(θ) do
                C, _ = evaluate(θ, full)
                return C
            end
        end

        function g!(G, θ)
            return _evaluate_or_explain(θ) do
                # `evaluate` gives dC/dŷ per sample and leaves the model at θ,
                # which is what turning it into dC/dθ needs
                _, dC = evaluate(θ, full; with_gradients = true)
                G .= _flat_parameter_gradient(model.forecast, dC, X)
                return G
            end
        end

        initial_sol = extract_params(model.forecast)
        lower = get(params, :lower_bounds, nothing)
        upper = get(params, :upper_bounds, nothing)
        bounded = !isnothing(lower) || !isnothing(upper)

        # `Optim` decides from the algorithm type whether it needs a gradient, so
        # only pass one when it can be used - handing `g!` to Nelder-Mead makes
        # `Optim` pick a gradient-based path it was not asked for
        gradient_based = _requires_gradient(algorithm)

        # if the solve throws, put the model back where it started. Otherwise it
        # is left at whatever the optimizer last tried, which is typically far
        # outside the sensible range - and then the remedy the error suggests,
        # retrying with bounds, fails too because the initial point is now
        # outside them
        res = try
            _optimize_with(
                fitness,
                g!,
                initial_sol,
                algorithm,
                optim_options,
                lower,
                upper,
                gradient_based,
                bounded,
            )
        catch
            apply_params(model.forecast, initial_sol)
            rethrow()
        end

        # update model parameters
        final_sol = Optim.minimizer(res)
        apply_params(model.forecast, final_sol)
        final_cost = Optim.minimum(res)
        return Solution(final_cost, final_sol)
    end
end

"""
    _optimize_with(fitness, g!, initial_sol, algorithm, options, lower, upper, gradient_based, bounded)

Dispatch to the right `Optim.optimize` signature.

Optim has four of them here, along two independent axes: whether a gradient is
supplied, and whether the search is boxed. The box-constrained call takes the
bounds positionally rather than through `Optim.Options`, and needs the algorithm
wrapped in `Fminbox`.
"""
function _optimize_with(
    fitness,
    g!,
    initial_sol,
    algorithm,
    optim_options,
    lower,
    upper,
    gradient_based::Bool,
    bounded::Bool,
)
    return if bounded
        # the box-constrained call takes a different signature entirely, and
        # `Fminbox` is what turns an unconstrained algorithm into a bounded one
        lo = isnothing(lower) ? fill(-Inf, length(initial_sol)) : lower
        hi = isnothing(upper) ? fill(Inf, length(initial_sol)) : upper
        inner =
            algorithm isa Optim.Fminbox ? algorithm : Optim.Fminbox(algorithm)
        if gradient_based
            Optim.optimize(
                fitness,
                g!,
                lo,
                hi,
                initial_sol,
                inner,
                optim_options,
            )
        else
            Optim.optimize(fitness, lo, hi, initial_sol, inner, optim_options)
        end
    elseif gradient_based
        Optim.optimize(fitness, g!, initial_sol, algorithm, optim_options)
    else
        Optim.optimize(fitness, initial_sol, algorithm, optim_options)
    end
end

"""
    _requires_gradient(algorithm)

Whether an Optim.jl algorithm consumes first-order information.

`Optim.FirstOrderOptimizer` covers the gradient-based methods (`GradientDescent`,
`BFGS`, `LBFGS`, `ConjugateGradient`, `MomentumGradientDescent`) and
`SecondOrderOptimizer` the Hessian-based ones, which also accept a gradient.
Everything else — `NelderMead`, `ParticleSwarm`, `SimulatedAnnealing`,
`BrentMethod` — is derivative-free.
"""
function _requires_gradient(algorithm)
    return algorithm isa Optim.FirstOrderOptimizer ||
           algorithm isa Optim.SecondOrderOptimizer
end

function _train!(
    ::Type{OptimMode},
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::Dict{Symbol,Any},
)
    return _train_with_optim!(model, X, Y, params, _optim_algorithm(params))
end

function _train!(
    ::Type{NelderMeadMode},
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::Dict{Symbol,Any},
)
    if haskey(params, :algorithm)
        throw(
            ArgumentError(
                "`NelderMeadMode` does not accept an `algorithm`; use " *
                "`Options(OptimMode; algorithm = ...)` instead.",
            ),
        )
    end
    # the two Nelder-Mead-specific keywords this mode has always accepted are
    # translated into the algorithm object that `OptimMode` takes directly
    algorithm = Optim.NelderMead(;
        initial_simplex = get(
            params,
            :initial_simplex,
            Optim.AffineSimplexer(),
        ),
        parameters = get(params, :parameters, Optim.AdaptiveParameters()),
    )
    return _train_with_optim!(model, X, Y, params, algorithm)
end

function _train!(
    ::Type{NelderMeadMPIMode},
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::Dict{Symbol,Any},
)
    return _train!(NelderMeadMode, model, X, Y, _as_mpi_params(params))
end

"""
    _train_with_nelder_mead!(model, X, Y, params)

Deprecated. Retained because it was reachable as
`ApplicationDrivenLearning._train_with_nelder_mead!`; use
[`_train_with_optim!`](@ref) or `train!` with [`OptimMode`](@ref).
"""
function _train_with_nelder_mead!(
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::Dict{Symbol,Any},
)
    return _train!(NelderMeadMode, model, X, Y, params)
end
