using Distributed

function compute_single_step_cost(
    model::Model,
    y::Vector{<:Real},
    yhat::Vector{<:Real},
)

    MOI.set.(
        model.plan, 
        POI.ParameterValue(), 
        model.plan_forecast_params, 
        yhat
    )
    optimize!(model.plan)
    fix.(assess_forecast_vars(model), y; force=true)
    set_normalized_rhs.(
        model.assess[:assess_policy_fix], 
        value.(plan_policy_vars(model))
    )
    optimize!(model.assess)
    return objective_value(model.assess)
end

"""
Computes the gradient of the cost function (C) with respect to the predictions (̂y).
"""
function compute_single_step_gradient(
    model::Model,
    dCdz::Vector{<:Real},
    dCdy::Vector{<:Real}
)               
    dCdz .= dual.(model.assess[:assess_policy_fix])
    for i=1:size(model.policy_vars, 1)
        MOI.set(
            model.plan,
            DiffOpt.ReverseVariablePrimal(),
            plan_policy_vars(model)[i],
            dCdz[i]
        )
    end
    DiffOpt.reverse_differentiate!(model.plan)
    for j=1:size(model.forecast_vars, 1)
        dCdy[j] = MOI.get(
            model.plan,
            POI.ReverseParameter(),
            model.plan_forecast_params[j]
        )
    end

    return dCdy
end

function compute_cost(
    model::Model,
    X::Matrix{<:Real},
    Y::Matrix{<:Real},
    with_gradients::Bool=false,
    n_workers::Int=1
)

    # data size assertions
    @assert size(X)[2] == model.forecast.input_size "Input size mismatch"
    @assert size(Y)[2] == model.forecast.output_size "Output size mismatch"

    # build model variables if necessary
    build(model)

    # init parameters
    C = 0
    T = size(Y)[1]
    dC = zeros(model.forecast.output_size)
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

    # main loop - sequential
    if n_workers == 1
        for t=1:T
            result = _compute_step(Y[t, :], Yhat[t, :])
            C += result[1]
            dC .+= result[2]
        end

    # main loop - parallel
    elseif n_workers > 1
        
        # add workers and import package on first call
        if nprocs() < n_workers
            error("Please add workers and import the AppDrivenLearning module @everywhere before calling `train!` function.")
        end

        # parallel computation
        result = pmap(_compute_step, [Y[t, :] for t=1:T], [Yhat[t, :] for t=1:T])
        C = sum([r[1] for r in result])
        dC = sum([r[2] for r in result])

    else
        error("Invalid number of workers")
    end

    if with_gradients
        return C, dC
    end
    return C

end
