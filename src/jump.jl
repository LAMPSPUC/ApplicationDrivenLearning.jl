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
    model::AppDrivenModel, 
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
    model::AppDrivenModel, 
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
function Plan(model::AppDrivenModel)
    return model.plan::JuMP.Model
end

function Assess(model::AppDrivenModel)
    return model.assess::JuMP.Model
end

# jump functions
function JuMP.objective_sense(model::AppDrivenModel)
    @assert JuMP.objective_sense(model.plan) == JuMP.objective_sense(model.assess)
    return JuMP.objective_sense(model.plan)
end

JuMP.num_variables(m::AppDrivenModel) = JuMP.num_variables(m.plan) + JuMP.num_variables(m.assess)

function JuMP.show_constraints_summary(io::IO, model::AppDrivenModel)
    println("Plan Model:")
    JuMP.show_constraints_summary(io, model.plan)
    println("\nAssess Model:")
    JuMP.show_constraints_summary(io, model.assess)
    return
end

function JuMP.show_backend_summary(io::IO, model::AppDrivenModel)
    println("Plan Model:")
    JuMP.show_backend_summary(io, model.plan)
    println("\nAssess Model:")
    JuMP.show_backend_summary(io, model.assess)
    return
end

JuMP.object_dictionary(model::AppDrivenModel) = model.obj_dict

function JuMP.set_optimizer(model::AppDrivenModel, builder)
    # set diffopt optimizer for plan model
    new_diff_optimizer = DiffOpt.diff_optimizer(builder)
    JuMP.set_optimizer(
        model.plan,
        () -> POI.Optimizer(new_diff_optimizer)
    )

    # basic setting for assess model
    JuMP.set_optimizer(model.assess, builder)
end

function JuMP.set_silent(model::AppDrivenModel)
    MOI.set(model.plan, MOI.Silent(), true)
    MOI.set(model.assess, MOI.Silent(), true)
end