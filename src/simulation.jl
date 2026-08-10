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
            MOI.set(model.plan, POI.ParameterValue(), params[j], yhat[j])
        end
    end
    # optimize plan model
    @timeit_debug _TIMER "plan_solve" optimize!(model.plan)
    # check for solution and fix assess policy vars
    try
        @timeit_debug _TIMER "fix_policy" begin
            cons = assess_policy_fix_cons(model)
            policy_vars = plan_policy_vars(model)
            for i in eachindex(cons, policy_vars)
                set_normalized_rhs(cons[i], value(policy_vars[i]))
            end
        end
    catch e
        println("Optimization failed for PLAN model.")
        throw(e)
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
    try
        return objective_value(model.assess)
    catch e
        println("Optimization failed for ASSESS model")
        throw(e)
    end
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
  - `X::Matrix{<:Real}`: input data.
  - `Y::Matrix{<:Real}`: true values.
  - `with_gradients::Bool=false`: flag to compute and return the cost gradients
    with respect to the forecasts. When set, the second returned value is the
    per-sample gradient matrix of size `(T, output_size)`.
  - `aggregate::Bool=true`: when true, the returned cost is averaged over the `T`
    samples. Only affects the cost; gradients are always per-sample.
    ...
"""
function compute_cost(
    model::Model,
    X::Matrix{<:Real},
    Y::Matrix{<:Real},
    with_gradients::Bool = false,
    aggregate::Bool = true,
)
    if isnothing(model.forecast)
        throw(
            ArgumentError(
                "No forecast model set. Call set_forecast_model first.",
            ),
        )
    end

    # data size assertions
    @assert size(X)[1] == size(Y)[1] "X and Y must have the same number of samples"
    @assert size(X)[2] == model.forecast.input_size "Input size mismatch"
    @assert size(Y)[2] == model.forecast.output_size "Output size mismatch"

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
    compute_cost(model, X, Y_dict, with_gradients=false, aggregate=true)

Variant of [`compute_cost`](@ref) that takes the realized values as a
dictionary mapping each [`Forecast`](@ref) variable to its series, instead of
a matrix. The columns are ordered to match the predictive model output.
"""
function compute_cost(
    model::Model,
    X::Matrix{<:Real},
    Y_dict::Dict{<:Forecast,<:Vector},
    with_gradients::Bool = false,
    aggregate::Bool = true,
)
    return compute_cost(
        model,
        X,
        _dict_to_var_indexed_matrix(Y_dict, model.forecast.output_variables),
        with_gradients,
        aggregate,
    )
end
