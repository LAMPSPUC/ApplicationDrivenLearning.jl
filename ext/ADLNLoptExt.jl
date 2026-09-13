"""
NLopt.jl backend for [`ApplicationDrivenLearning.NLoptMode`](@ref).

Loaded automatically once `NLopt` is available. The mode type lives in the core
package so that using it without `using NLopt` fails with a readable message
instead of an `UndefVarError`; only the trainer is here.
"""
module ADLNLoptExt

import ApplicationDrivenLearning as ADL
import NLopt

# keywords consumed here rather than set on the `Opt` object
const _NLOPT_RESERVED = (:algorithm, :parallel, :mpi_finalize)

# NLopt return codes that mean the solve went wrong, as opposed to the ones that
# just report which stopping criterion fired
const _NLOPT_FAILURES =
    (:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :ROUNDOFF_LIMITED, :FORCED_STOP)

"""
    _is_gradient_algorithm(algorithm::Symbol)

Whether an NLopt algorithm consumes first-order information, which its name
encodes: the second letter is `D` for derivative-based and `N` for
derivative-free (`:LD_LBFGS` versus `:LN_BOBYQA`).
"""
function _is_gradient_algorithm(algorithm::Symbol)
    name = String(algorithm)
    return length(name) >= 2 && name[2] == 'D'
end

function ADL._train!(
    ::Type{ADL.NLoptMode},
    model::ADL.Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    params::Dict{Symbol,Any},
)
    algorithm = get(params, :algorithm, :LN_NELDERMEAD)
    θ0 = ADL.extract_params(model.forecast)

    opt = NLopt.Opt(algorithm, length(θ0))
    # every remaining keyword is an NLopt option, named as NLopt names it
    for (key, value) in params
        key in _NLOPT_RESERVED && continue
        setproperty!(opt, key, value)
    end

    needs_gradient = _is_gradient_algorithm(algorithm)

    # the parallel backend is handled entirely by the core: this extension gets an
    # `evaluate` closure and never touches MPI, so `parallel = MPIBackend()` works
    # here with no MPI-specific code of its own
    return ADL._with_evaluator(
        ADL._parallel_backend(params),
        model,
        X,
        Y,
    ) do evaluate
        # every evaluation here covers the whole dataset
        full = ADL._full_batch(X, Y)

        opt.min_objective = function (x, grad)
            return ADL._evaluate_or_explain(x) do
                # NLopt passes an empty `grad` when the algorithm does not want
                # one, and computing gradients we then discard would mean a
                # wasted reverse pass through DiffOpt on every evaluation
                if needs_gradient && length(grad) > 0
                    # the cost comes from the same evaluation as the gradient:
                    # recomputing it would double the work and, under MPI, would
                    # do it on the controller instead of across the workers
                    C, dC = evaluate(x, full; with_gradients = true)
                    grad .=
                        ADL._flat_parameter_gradient(model.forecast, dC, X)
                    return C
                end
                C, _ = evaluate(x, full)
                return C
            end
        end

        # on failure, put the model back where it started: otherwise it is left
        # at whatever the optimizer last tried, and retrying with the bounds the
        # error suggests would then fail for having an initial point outside them
        final_cost, final_sol, ret = try
            NLopt.optimize(opt, θ0)
        catch
            ADL.apply_params(model.forecast, θ0)
            rethrow()
        end

        if ret in _NLOPT_FAILURES
            @warn "NLopt did not converge" algorithm return_code = ret
        end

        # NLopt's returned minimizer is not necessarily the last point evaluated,
        # so the model has to be put back onto it explicitly
        ADL.apply_params(model.forecast, final_sol)
        return ADL.Solution(final_cost, final_sol)
    end
end

end
