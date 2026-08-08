"""
    compute_single_step_cost(model::Model, y::Vector{<:Real}, yhat::Vector{<:Real})

Evaluate the assessed cost of a single sample.

The prediction `yhat` is written into the plan model's forecast parameters and
the plan model is solved; the resulting policy is then fixed in the assess
model, whose forecast variables are fixed to the realized values `y`. The
optimal objective of the assess model is returned.

Requires [`build`](@ref) to have been called on `model`.
"""
function compute_single_step_cost(
    model::Model,
    y::Vector{<:Real},
    yhat::Vector{<:Real},
)
    # set forecast params as prediction output
    MOI.set.(model.plan, POI.ParameterValue(), model.plan_forecast_params, yhat)
    # optimize plan model
    optimize!(model.plan)
    # check for solution and fix assess policy vars
    try
        set_normalized_rhs.(
            model.assess[:assess_policy_fix],
            value.(plan_policy_vars(model)),
        )
    catch e
        println("Optimization failed for PLAN model.")
        throw(e)
    end
    # fix assess forecast vars on observer values
    fix.(assess_forecast_vars(model), y; force = true)
    # optimize assess model
    optimize!(model.assess)
    # check for optimization
    try
        return objective_value(model.assess)
    catch e
        println("Optimization failed for ASSESS model")
        throw(e)
    end
end

"""
    compute_single_step_gradient(model::Model, dCdz::Vector{<:Real}, dCdy::Vector{<:Real})

Compute the gradient of the assessed cost `C` with respect to the predictions
`ŷ` for the sample most recently evaluated by
[`compute_single_step_cost`](@ref).

The duals of the `assess_policy_fix` constraint give `dC/dz`, the sensitivity
with respect to the policy; DiffOpt then propagates them backwards through the
plan model to the forecast parameters.

Both `dCdz` and `dCdy` are overwritten in place and `dCdy` is returned. Note
that the returned vector aliases the `dCdy` argument, so callers that keep the
result across several samples must copy it.
"""
function compute_single_step_gradient(
    model::Model,
    dCdz::Vector{<:Real},
    dCdy::Vector{<:Real},
)
    dCdz .= dual.(model.assess[:assess_policy_fix])
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
    DiffOpt.reverse_differentiate!(model.plan)
    for j = 1:size(model.forecast_vars, 1)
        dCdy[j] =
            MOI.get(
                model.plan,
                DiffOpt.ReverseConstraintSet(),
                ParameterRef(model.plan_forecast_params[j]),
            ).value
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
    build(model)

    # init parameters
    T = size(Y)[1]
    C = zeros(T)
    dC = zeros((T, model.forecast.output_size))
    # solver duals and DiffOpt sensitivities are Float64, so the gradient
    # buffers must be too - narrowing them here would silently round the
    # gradients to single precision
    dCdz = Vector{Float64}(undef, length(model.policy_vars))
    dCdy = Vector{Float64}(undef, model.forecast.output_size)

    function _compute_step(y, yhat)
        c = compute_single_step_cost(model, y, yhat)
        if with_gradients
            dc = compute_single_step_gradient(model, dCdz, dCdy)
            return c, dc
        end
        return c, 0
    end

    # get predictions
    Yhat = model.forecast(X')'  # size=(T, output_size)

    # main loop to compute cost
    for t = 1:T
        result = _compute_step(Y[t, :], Yhat[t, :])
        C[t] += result[1]
        dC[t, :] .+= result[2]
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
        dict_to_var_indexed_matrix(Y_dict, model.forecast.output_variables),
        with_gradients,
        aggregate,
    )
end
