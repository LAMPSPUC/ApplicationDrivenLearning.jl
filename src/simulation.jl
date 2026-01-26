function compute_single_step_cost(
    model::Model,
    y::VariableIndexedVector,
    yhat::VariableIndexedVector,
)
    # set forecast params as prediction output
    MOI.set.(
        model.plan,
        POI.ParameterValue(),
        model.plan_forecast_params,
        yhat[model.forecast_vars].data,
    )
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
    fix.(model.forecast_vars.assess, y[model.forecast_vars].data; force = true)
    # optimize assess model
    optimize!(model.assess)
    # check for optimization
    try
        return objective_value(model.assess)
    catch e
        println("Optimization failed for ASSESS model")
        throw(e)
    end
    # should never get here
    return 0
end

"""
Computes the gradient of the cost function (C) with respect to the predictions (̂y).
"""
function compute_single_step_gradient(
    model::Model,
    dCdz::Vector{<:Real},
    dCdy::VariableIndexedVector{<:Real},
)
    dCdz .= dual.(model.assess[:assess_policy_fix])
    DiffOpt.empty_input_sensitivities!(model.plan)
    for i = 1:size(model.policy_vars, 1)
        MOI.set(
            model.plan,
            DiffOpt.ReverseVariablePrimal(),
            plan_policy_vars(model)[i],
            dCdz[i],
        )
    end
    DiffOpt.reverse_differentiate!(model.plan)
    for fv in model.forecast_vars
        dCdy[fv] =
            MOI.get(
                model.plan,
                DiffOpt.ReverseConstraintSet(),
                ParameterRef(fv.plan),
            ).value
    end

    return dCdy
end

"""
    compute_cost(model, X, Y, with_gradients=false)

Compute the cost function (C) based on the model predictions and the true values.

...

# Arguments

  - `model::ApplicationDrivenLearning.Model`: model to evaluate.
  - `X::Matrix{<:Real}`: input data.
  - `Y::Matrix{<:Real}`: true values.
  - `with_gradients::Bool=false`: flag to compute and return gradients.
    ...
"""
function compute_cost(
    model::Model,
    X::Matrix{<:Real},
    Y::Dict{<:Forecast,<:Vector},
    with_gradients::Bool = false,
    aggregate::Bool = true,
)

    # data size assertions
    @assert size(X)[2] == model.forecast.input_size "Input size mismatch"
    @assert length(Y) == model.forecast.output_size "Output size mismatch"

    # build model variables if necessary
    build(model)

    # init parameters
    T = length.(collect(values(Y)))[1]
    C = Vector{Float32}(undef, T)
    dCdz = Vector{Float32}(undef, size(model.policy_vars, 1))
    dCdy = VariableIndexedVector{Float32}(undef, model.forecast_vars)
    dC = VariableIndexedMatrix{Float32}(undef, model.forecast_vars, T)

    function _get_index_y(Y::Dict{<:Forecast,<:Vector}, idx::Int)
        var_index = Vector{Forecast}(undef, model.forecast.output_size)
        y_values = Vector{Real}(undef, model.forecast.output_size)
        for (i, (fvar, vals)) in enumerate(Y)
            var_index[i] = fvar
            y_values[i] = vals[idx]
        end
        return VariableIndexedVector(y_values, var_index)
    end

    function _compute_step(
        y::VariableIndexedVector,
        yhat::VariableIndexedVector,
    )
        c = compute_single_step_cost(model, y, yhat)
        if with_gradients
            dc = compute_single_step_gradient(model, dCdz, dCdy)
            return c, dc
        end
        return c, 0
    end

    # get predictions
    Yhat = model.forecast(X')  # size=(output_size, T) -> VariableIndexedMatrix

    # main loop to compute cost
    for t = 1:T
        result = _compute_step(_get_index_y(Y, t), Yhat[t])
        C[t] = result[1]
        dC[t] = result[2]
    end

    # aggregate cost if requested
    if aggregate
        C = sum(C) / T
        dC = sum(dC) / T
    end

    if with_gradients
        return C, dC
    end
    return C
end
