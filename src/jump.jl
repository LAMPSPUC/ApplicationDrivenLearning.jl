"""
    _PolicyInfo

Internal container produced by `JuMP.build_variable` when a variable is
declared with the [`Policy`](@ref) type. It carries the `JuMP.VariableInfo`
used to create the twin variables in the plan and assess models.
"""
struct _PolicyInfo
    plan::JuMP.VariableInfo
    assess::JuMP.VariableInfo
    kwargs::Any
end

function JuMP.build_variable(
    _error::Function,
    info::JuMP.VariableInfo,
    ::Type{Policy};
    kwargs...,
)
    return _PolicyInfo(info, info, kwargs)
end

function JuMP.add_variable(model::Model, policy_info::_PolicyInfo, name::String)
    policy = Policy(
        JuMP.add_variable(
            model.plan,
            JuMP.ScalarVariable(policy_info.plan),
            name * "_plan",
        ),
        JuMP.add_variable(
            model.assess,
            JuMP.ScalarVariable(policy_info.assess),
            name * "_assess",
        ),
    )
    push!(model.policy_vars, policy)
    push!(model._plan_policy_vars, policy.plan)
    push!(model._assess_policy_vars, policy.assess)
    return policy
end

"""
    _ForecastInfo

Internal container produced by `JuMP.build_variable` when a variable is
declared with the [`Forecast`](@ref) type. It carries the
`JuMP.VariableInfo` used to create the twin variables in the plan and assess
models. Bounds are not supported on forecast variables and are dropped with a
warning.
"""
struct _ForecastInfo
    plan::JuMP.VariableInfo
    assess::JuMP.VariableInfo
    kwargs::Any
end

function JuMP.build_variable(
    _error::Function,
    info::JuMP.VariableInfo,
    ::Type{Forecast};
    kwargs...,
)
    return _ForecastInfo(info, info, kwargs)
end

function JuMP.add_variable(
    model::Model,
    forecast_info::_ForecastInfo,
    name::String,
)
    forecast = Forecast(
        JuMP.add_variable(
            model.plan,
            JuMP.ScalarVariable(forecast_info.plan),
            name * "_plan",
        ),
        JuMP.add_variable(
            model.assess,
            JuMP.ScalarVariable(forecast_info.assess),
            name * "_assess",
        ),
    )

    # forecast variables can't have bounds
    if JuMP.has_lower_bound(forecast.plan)
        @warn "Forecast variable lower bound will be removed."
        JuMP.delete_lower_bound(forecast.plan)
        JuMP.delete_lower_bound(forecast.assess)
    end

    if JuMP.has_upper_bound(forecast.plan)
        @warn "Forecast variable upper bound will be removed."
        JuMP.delete_upper_bound(forecast.plan)
        JuMP.delete_upper_bound(forecast.assess)
    end

    push!(model.forecast_vars, forecast)
    push!(model._plan_forecast_vars, forecast.plan)
    push!(model._assess_forecast_vars, forecast.assess)
    return forecast
end

"""
    Plan(model::ApplicationDrivenLearning.Model)

Return the inner `JuMP.Model` used for planning, i.e. the problem solved with
the predictive model output in place of the [`Forecast`](@ref) variables. Use
it as the target of the `@variable`, `@constraint` and `@objective` macros.
"""
function Plan(model::Model)
    return model.plan::JuMP.Model
end

"""
    Assess(model::ApplicationDrivenLearning.Model)

Return the inner `JuMP.Model` used for assessment, i.e. the problem solved
with the realized values in place of the [`Forecast`](@ref) variables and the
[`Policy`](@ref) variables fixed to the plan decision. Use it as the target of
the `@variable`, `@constraint` and `@objective` macros.
"""
function Assess(model::Model)
    return model.assess::JuMP.Model
end

# jump functions
function JuMP.objective_sense(model::Model)
    @assert JuMP.objective_sense(model.plan) ==
            JuMP.objective_sense(model.assess)
    return JuMP.objective_sense(model.plan)
end

function JuMP.num_variables(m::Model)
    return JuMP.num_variables(m.plan) + JuMP.num_variables(m.assess)
end

function JuMP.show_constraints_summary(io::IO, model::Model)
    println(io, "Plan Model:")
    JuMP.show_constraints_summary(io, model.plan)
    println(io, "\nAssess Model:")
    JuMP.show_constraints_summary(io, model.assess)
    return
end

function JuMP.show_backend_summary(io::IO, model::Model)
    println(io, "Plan Model:")
    JuMP.show_backend_summary(io, model.plan)
    println(io, "\nAssess Model:")
    JuMP.show_backend_summary(io, model.assess)
    return
end

JuMP.object_dictionary(model::Model) = model.obj_dict

"""
    JuMP.set_optimizer(model::ApplicationDrivenLearning.Model, builder)

Set the solver used by both inner models. The plan model is wrapped in a
`DiffOpt.diff_optimizer`, so that it can be differentiated with respect to the
forecast parameters; the assess model uses `builder` directly.
"""
function JuMP.set_optimizer(model::Model, builder)
    # set diffopt optimizer for plan model
    new_diff_optimizer = DiffOpt.diff_optimizer(builder)
    JuMP.set_optimizer(model.plan, () -> new_diff_optimizer)

    # basic setting for assess model
    JuMP.set_optimizer(model.assess, builder)

    return nothing
end

function JuMP.set_silent(model::Model)
    JuMP.set_silent(model.plan)
    JuMP.set_silent(model.assess)
    return
end

function JuMP.num_constraints(
    model::Model;
    count_variable_in_set_constraints::Bool,
)
    return JuMP.num_constraints(
        model.plan;
        count_variable_in_set_constraints = count_variable_in_set_constraints,
    ) + JuMP.num_constraints(
        model.assess;
        count_variable_in_set_constraints = count_variable_in_set_constraints,
    )
end

function Base.print(io::IO, model::Model)
    println(io, "Plan Model:")
    println(io, model.plan)
    println(io, "\nAssess Model:")
    println(io, model.assess)
    println(io, "\nForecast Model:")
    if isnothing(model.forecast)
        println(io, "Not defined.")
    else
        for network in model.forecast.networks
            println(io, network)
        end
    end
    return
end
