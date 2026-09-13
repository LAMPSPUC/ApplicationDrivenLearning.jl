"""
    _assert_has_solution(jump_model::JuMP.Model, what::String)

Throw unless `jump_model` holds a primal solution that can be read.

A *feasibility* check rather than a termination-status one on purpose. What the
caller needs is a point to read — the policy values out of the plan model, the
objective out of the assess model — and `has_values` asks exactly that.
Whitelisting termination statuses would ask something narrower and get it wrong
in both directions: it would reject a solution that hit an iteration or time
limit with a perfectly usable incumbent, and it would still have to consult the
primal status anyway before reading values.
"""
function _assert_has_solution(jump_model::JuMP.Model, what::String)
    status = termination_status(jump_model)
    reason = if status == MOI.INFEASIBLE
        "has no feasible solution"
    elseif status == MOI.DUAL_INFEASIBLE
        # MOI reports an unbounded primal as an infeasible dual, which reads as
        # the opposite of what it means unless it is spelled out
        "is unbounded, so no policy is optimal"
    elseif status == MOI.INFEASIBLE_OR_UNBOUNDED
        "is infeasible or unbounded; the solver could not tell which, which " *
        "presolve often causes - disabling it will say"
    elseif !has_values(jump_model)
        "has no solution to read"
    else
        return nothing
    end
    throw(
        ErrorException(
            "The $what model $reason: the solver returned termination status " *
            "$status and primal status $(primal_status(jump_model)). The " *
            "application cannot be evaluated at this prediction.",
        ),
    )
end

"""
    _compute_single_step_cost(model::Model, y::AbstractVector{<:Real}, yhat::AbstractVector{<:Real})

Evaluate the assessed cost of a single sample.

The prediction `yhat` is written into the plan model's forecast parameters and
the plan model is solved; the resulting policy is then fixed in the assess
model, whose forecast variables are fixed to the realized values `y`. The
optimal objective of the assess model is returned.

Requires `_build` to have been called on `model`.
"""
function _compute_single_step_cost(
    model::Model,
    y::AbstractVector{<:Real},
    yhat::AbstractVector{<:Real},
)
    # set forecast params as prediction output
    @timeit_debug _TIMER "set_params" begin
        params = model.plan_forecast_params
        for j in eachindex(params, yhat)
            set_parameter_value(params[j], yhat[j])
        end
    end
    # optimize plan model
    @timeit_debug _TIMER "plan_solve" optimize!(model.plan)
    # check for solution and fix assess policy vars
    _assert_has_solution(model.plan, "plan")
    @timeit_debug _TIMER "fix_policy" begin
        cons = assess_policy_fix_cons(model)
        policy_vars = plan_policy_vars(model)
        for i in eachindex(cons, policy_vars)
            set_normalized_rhs(cons[i], value(policy_vars[i])) # TODO: parameters?
        end
    end
    # fix assess forecast vars on observer values
    @timeit_debug _TIMER "fix_forecast" begin
        forecast_vars = assess_forecast_vars(model)
        for j in eachindex(forecast_vars, y)
            fix(forecast_vars[j], y[j]; force = true)
        end
    end
    # optimize assess model
    @timeit_debug _TIMER "assess_solve" optimize!(model.assess)
    # check for optimization
    _assert_has_solution(model.assess, "assess")
    return objective_value(model.assess)
end

"""
    _compute_single_step_gradient(model::Model, dCdz::Vector{<:Real}, dCdy::Vector{<:Real})

Compute the gradient of the assessed cost `C` with respect to the predictions
`ŷ` for the sample most recently evaluated by
`_compute_single_step_cost`.

The duals of the `assess_policy_fix` constraint give `dC/dz`, the sensitivity
with respect to the policy; DiffOpt then propagates them backwards through the
plan model to the forecast parameters.

Both `dCdz` and `dCdy` are overwritten in place and `dCdy` is returned. Note
that the returned vector aliases the `dCdy` argument, so callers that keep the
result across several samples must copy it.
"""
function _compute_single_step_gradient(
    model::Model,
    dCdz::Vector{<:Real},
    dCdy::Vector{<:Real},
)
    @timeit_debug _TIMER "read_duals" begin
        cons = assess_policy_fix_cons(model)
        for i in eachindex(dCdz, cons)
            dCdz[i] = dual(cons[i])
        end
    end
    @timeit_debug _TIMER "set_reverse_seed" begin
        DiffOpt.empty_input_sensitivities!(model.plan)
        policy_vars = plan_policy_vars(model)
        for i in eachindex(policy_vars)
            MOI.set(
                model.plan,
                DiffOpt.ReverseVariablePrimal(),
                policy_vars[i],
                dCdz[i],
            )
        end
    end
    @timeit_debug _TIMER "diffopt_reverse" DiffOpt.reverse_differentiate!(
        model.plan,
    )
    @timeit_debug _TIMER "read_sensitivities" begin
        for j = 1:size(model.forecast_vars, 1)
            dCdy[j] =
                MOI.get(
                    model.plan,
                    DiffOpt.ReverseConstraintSet(),
                    ParameterRef(model.plan_forecast_params[j]),
                ).value
        end
    end

    return dCdy
end

"""
    compute_cost(model, X, Y, with_gradients=false)

Compute the cost function (C) based on the model predictions and the true values matrix.

...

# Arguments

  - `model::ApplicationDrivenLearning.Model`: model to evaluate.
  - `X`: input data of size `(T, input_size)`. A matrix or vector, whose columns
    are taken in order, or a Tables.jl-compatible table such as a `DataFrame`,
    whose columns are selected by the predictive model's `input_names`.
  - `Y`: realized values. A `(T, output_size)` matrix, a vector, a
    Tables.jl-compatible table, or a `Dict` mapping each [`Forecast`](@ref)
    variable to its series. Table columns are matched to the forecast variables
    by name, never by position; see [`set_forecast_model`](@ref) for how those
    names are declared.
  - `with_gradients::Bool=false`: flag to compute and return the cost gradients
    with respect to the forecasts. When set, the second returned value is the
    per-sample gradient matrix of size `(T, output_size)`.
  - `aggregate::Bool=true`: when true, the returned cost is averaged over the `T`
    samples. Only affects the cost; gradients are always per-sample.
    ...
"""
function compute_cost(
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    with_gradients::Bool = false,
    aggregate::Bool = true,
)
    _assert_forecast_model_set(model)

    # data shape checks. `ArgumentError` rather than `@assert`: these validate
    # caller-supplied data, and an `@assert` is documented as removable at some
    # optimization levels, which would turn a wrong shape into a wrong answer
    if size(X, 1) != size(Y, 1)
        throw(
            ArgumentError(
                "`X` has $(size(X, 1)) sample(s) but `Y` has $(size(Y, 1)); " *
                "they must have the same number of rows.",
            ),
        )
    elseif size(X, 2) != model.forecast.input_size
        throw(
            ArgumentError(
                "`X` has $(size(X, 2)) column(s) but the predictive model takes " *
                "$(model.forecast.input_size) input(s).",
            ),
        )
    elseif size(Y, 2) != model.forecast.output_size
        throw(
            ArgumentError(
                "`Y` has $(size(Y, 2)) column(s) but the predictive model " *
                "produces $(model.forecast.output_size) output(s).",
            ),
        )
    end

    # build model variables if necessary
    _build(model)

    # init parameters
    T = size(Y)[1]
    C = zeros(T)
    dC = zeros((T, model.forecast.output_size))
    # solver duals and DiffOpt sensitivities are Float64, so the gradient
    # buffers must be too - narrowing them here would silently round the
    # gradients to single precision
    dCdz = Vector{Float64}(undef, length(model.policy_vars))
    dCdy = Vector{Float64}(undef, model.forecast.output_size)

    # get predictions; kept in the (output_size, T) layout the predictive model
    # produces, so that the per-sample prediction below is a contiguous column
    Yhat = @timeit_debug _TIMER "forward_pass" model.forecast(X')

    # main loop to compute cost - the two branches are written out so that the
    # gradient result has a single concrete type and so that the non-gradient
    # case does not touch `dC` at all
    @timeit_debug _TIMER "sample_loop" if with_gradients
        for t = 1:T
            C[t] += _compute_single_step_cost(
                model,
                view(Y, t, :),
                view(Yhat, :, t),
            )
            dc = _compute_single_step_gradient(model, dCdz, dCdy)
            for j in eachindex(dc)
                dC[t, j] += dc[j]
            end
        end
    else
        for t = 1:T
            C[t] += _compute_single_step_cost(
                model,
                view(Y, t, :),
                view(Yhat, :, t),
            )
        end
    end

    # aggregate cost if requested
    if aggregate
        C = sum(C) / T
    end

    if with_gradients
        return C, dC
    end
    return C
end

"""
    compute_cost(model, X, Y, with_gradients=false, aggregate=true)

Variant of [`compute_cost`](@ref) accepting any combination of the supported
input containers — matrices, vectors, Tables.jl-compatible tables such as a
`DataFrame`, and a `Dict` mapping each [`Forecast`](@ref) variable to its series.
Both arguments are normalized to matrices together — which is what lets the rows
be matched by `sample_key` — and the call is forwarded to the matrix method.

`X` and `Y` are `@nospecialize`d. Without it this wrapper is specialized on the
concrete container types, and specializing it drags the whole `compute_cost`
body — sample loop, solver calls and all — through inference again in the new
context: measured at **174 s** for the first table-typed call, against 0.001 s
once compiled. Nothing here benefits from specialization, since both arguments
are immediately converted to matrices.
"""
function compute_cost(
    model::Model,
    @nospecialize(X),
    @nospecialize(Y),
    with_gradients::Bool = false,
    aggregate::Bool = true,
)
    # reading the containers needs the forecast variables and the declared
    # schema, so the forecast model has to be set before that can happen at all
    _assert_forecast_model_set(model)
    Xm, Ym = _to_matrices(X, Y, model.forecast)
    return compute_cost(model, Xm, Ym, with_gradients, aggregate)
end

"""
    _evaluate_or_explain(f, θ)

Run one objective evaluation, and if it fails, say what the optimizer was doing
when it did.

An external optimizer explores freely, and nothing stops it proposing parameters
whose predictions the application cannot accommodate — a negative forecast where
the plan model needs a non-negative one, say. The plan model is then infeasible
and the failure surfaces from deep inside the solver stack (typically DiffOpt
reporting `termination status INFEASIBLE`), naming neither the parameters that
caused it nor anything the caller can act on. This adds both.
"""
function _evaluate_or_explain(f, θ)
    try
        return f()
    catch err
        throw(
            ErrorException(
                "Evaluating the application at θ = $(_short_vector(θ)) failed:\n" *
                sprint(showerror, err) *
                "\n\nThis usually means the optimizer proposed parameters whose " *
                "predictions the application cannot accommodate, leaving the " *
                "region where the plan model is feasible. Constrain the search " *
                "with `lower_bounds` / `upper_bounds`, or make the application " *
                "feasible for every prediction it can be given.",
            ),
        )
    end
end

"""
    _short_vector(θ)

A representation of `θ` that stays readable for a network with many parameters.
"""
function _short_vector(θ)
    length(θ) <= 8 && return repr(θ)
    lo, hi = extrema(θ)
    return "$(length(θ))-element vector with extrema ($lo, $hi)"
end
