# policy variable type
struct PolicyInfo
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
    return PolicyInfo(
        info,
        info,
        kwargs
    )
end

function JuMP.add_variable(
    model::Model, 
    policy_info::PolicyInfo, 
    name::String
)
    policy = Policy(
        JuMP.add_variable(
            model.plan, 
            JuMP.ScalarVariable(policy_info.plan), 
            name * "_plan"
        ),
        JuMP.add_variable(
            model.assess, 
            JuMP.ScalarVariable(policy_info.assess), 
            name * "_assess"
        )
    )
    push!(model.policy_vars, policy)
    return policy
end

# forecast variable type
struct ForecastInfo
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
    return ForecastInfo(
        info,
        info,
        kwargs
    )
end

function JuMP.add_variable(
    model::Model, 
    forecast_info::ForecastInfo, 
    name::String
)
    forecast = Forecast(
        JuMP.add_variable(
            model.plan, 
            JuMP.ScalarVariable(forecast_info.plan), 
            name * "_plan"
        ),
        JuMP.add_variable(
            model.assess, 
            JuMP.ScalarVariable(forecast_info.assess), 
            name * "_assess"
        )
    )
    push!(model.forecast_vars, forecast)
    return forecast
end

# plan and assess models
function Plan(model::Model)
    return model.plan::JuMP.Model
end

function Assess(model::Model)
    return model.assess::JuMP.Model
end

# jump functions
function JuMP.objective_sense(model::Model)
    @assert JuMP.objective_sense(model.plan) == JuMP.objective_sense(model.assess)
    return JuMP.objective_sense(model.plan)
end

JuMP.num_variables(m::Model) = JuMP.num_variables(m.plan) + JuMP.num_variables(m.assess)

function JuMP.show_constraints_summary(io::IO, model::Model)
    println("Plan Model:")
    JuMP.show_constraints_summary(io, model.plan)
    println("\nAssess Model:")
    JuMP.show_constraints_summary(io, model.assess)
    return
end

function JuMP.show_backend_summary(io::IO, model::Model)
    println("Plan Model:")
    JuMP.show_backend_summary(io, model.plan)
    println("\nAssess Model:")
    JuMP.show_backend_summary(io, model.assess)
    return
end

JuMP.object_dictionary(model::Model) = model.obj_dict

function JuMP.set_optimizer(model::Model, builder, evaluate_duals::Bool=true)
    # set diffopt optimizer for plan model
    new_diff_optimizer = DiffOpt.diff_optimizer(builder)
    JuMP.set_optimizer(
        model.plan,
        () -> POI.Optimizer(new_diff_optimizer; evaluate_duals=evaluate_duals)
    )

    # basic setting for assess model
    JuMP.set_optimizer(model.assess, builder)
end

function JuMP.set_silent(model::Model)
    MOI.set(model.plan, MOI.Silent(), true)
    MOI.set(model.assess, MOI.Silent(), true)
end