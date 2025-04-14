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
    # should never get here
    return 0
end

"""
Computes the gradient of the cost function (C) with respect to the predictions (̂y).
"""
function compute_single_step_gradient(
    model::Model,
    dCdz::Vector{<:Real},
    dCdy::Vector{<:Real},
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
    Y::Matrix{<:Real},
    with_gradients::Bool = false,
    aggregate::Bool = true,
)

    # data size assertions
    @assert size(X)[2] == model.forecast.input_size "Input size mismatch"
    @assert size(Y)[2] == model.forecast.output_size "Output size mismatch"

    # build model variables if necessary
    build(model)

    # init parameters
    T = size(Y)[1]
    C = zeros(T)
    dC = zeros((T, model.forecast.output_size))
    dCdz = Vector{Float32}(undef, size(model.policy_vars, 1))
    dCdy = Vector{Float32}(undef, model.forecast.output_size)

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
        dC = sum(dC, dims = 1)[1, :] / T
    end

    if with_gradients
        return C, dC
    end
    return C
end
