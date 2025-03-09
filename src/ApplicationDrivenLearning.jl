module ApplicationDrivenLearning

using Flux
using JuMP
using DiffOpt
import ParametricOptInterface as POI
import Base.*, Base.+

include("flux_utils.jl")
include("predictive_model.jl")

"""
    Policy{T}

Policy variable type that holds plan and assess variables.
"""
struct Policy{T}
    plan::T
    assess::T
end

+(p1::Policy, p2::Policy) = Policy(p1.plan + p2.plan, p1.assess + p2.assess)
*(c::Number, p::Policy) = Policy(c * p.plan, c * p.assess)

"""
    Forecast{T}

Forecast variable type that holds plan and assess variables.
"""
struct Forecast{T}
    plan::T
    assess::T
end

function +(p1::Forecast, p2::Forecast)
    return Forecast(p1.plan + p2.plan, p1.assess + p2.assess)
end
*(c::Number, p::Forecast) = Forecast(c * p.plan, c * p.assess)

"""
    Model <: JuMP.AbstractModel

Create an empty ApplicationDrivenLearning.Model with empty plan and assess
models, missing forecast model and default settings.
"""
mutable struct Model <: JuMP.AbstractModel
    plan::JuMP.Model
    assess::JuMP.Model
    forecast::Union{PredictiveModel,Nothing}

    # variable arrays
    policy_vars::Vector{Policy}
    forecast_vars::Vector{Forecast}
    plan_forecast_params::Vector{JuMP.VariableRef}

    # API part
    obj_dict::Dict{Symbol,Any}
    build::Bool

    function Model()
        plan = JuMP.Model()
        assess = JuMP.Model()

        return new(
            plan,
            assess,
            nothing,
            Vector{Policy}(),
            Vector{Forecast}(),
            Vector{JuMP.VariableRef}(),
            Dict{Symbol,Any}(),
            false,
        )
    end
end

"""
Returns vector of policy variables from plan model.
"""
function plan_policy_vars(model::Model)
    return [v.plan for v in model.policy_vars]
end

"""
Returns vector of policy variables from assess model.
"""
function assess_policy_vars(model::Model)
    return [v.assess for v in model.policy_vars]
end

"""
Returns vector of forecast variables from plan model.
"""
function plan_forecast_vars(model::Model)
    return [v.plan for v in model.forecast_vars]
end

"""
Returns vector of forecast variables from assess model.
"""
function assess_forecast_vars(model::Model)
    return [v.assess for v in model.forecast_vars]
end

"""
Sets Chain, Dense or custom PredictiveModel object as
forecast model.
"""
function set_forecast_model(
    model::Model,
    network::Union{PredictiveModel,Flux.Chain,Flux.Dense},
)
    if typeof(network) == PredictiveModel
        forecast = network
    else
        forecast = PredictiveModel(network)
    end
    @assert forecast.output_size == size(model.forecast_vars, 1)
    return model.forecast = forecast
end

"""
    forecast(model, X)

Return forecast model output for given input.
"""
function forecast(model::Model, X::AbstractMatrix)
    return model.forecast(X)
end

"""
Creates new forecast variables to plan model using MOI.Parameter
and new constraint fixing to original forecast variables.
"""
function build_plan_model_forecast_params(model::Model)
    # adds parametrized forecast variables using MOI.Parameter
    forecast_size = size(model.forecast_vars)[1]
    model.plan_forecast_params = @variable(
        model.plan,
        _forecast[1:forecast_size] in MOI.Parameter.(zeros(forecast_size))
    )
    # fixes old and new prediction variables together
    @constraint(
        model.plan,
        plan_forecast_fix,
        model.plan_forecast_params .== plan_forecast_vars(model)
    )
end

"""
Creates new constraint to assess model that fixes policy variables.
"""
function build_assess_model_policy_constraint(model::Model)
    @constraint(
        model.assess,
        assess_policy_fix,
        assess_policy_vars(model) .== 0
    )
end

"""
Calls functions that set new variables and constraints that are
necessary to cost computation.
"""
function build(model::Model)
    if model.build
        return
    end
    model.build = true

    # build plan model
    build_plan_model_forecast_params(model)
    return build_assess_model_policy_constraint(model)
end

include("jump.jl")
include("simulation.jl")
include("options.jl")
include("solution.jl")
include("optimizers/gradient.jl")
include("optimizers/nelder_mead.jl")
include("optimizers/nelder_mead_mpi.jl")
include("optimizers/gradient_mpi.jl")
include("optimizers/bilevel.jl")

"""
    train!(model, X, y, options)

Train model using given data and options.
"""
function train!(
    model::Model,
    X::Matrix{<:Real},
    y::Matrix{<:Real},
    options::Options,
)
    if options.mode == NelderMeadMode
        return train_with_nelder_mead!(model, X, y, options.params)
    elseif options.mode == GradientMode
        return train_with_gradient!(model, X, y, options.params)
    elseif options.mode == NelderMeadMPIMode
        return train_with_nelder_mead_mpi!(model, X, y, options.params)
    elseif options.mode == GradientMPIMode
        return train_with_gradient_mpi!(model, X, y, options.params)
    elseif options.mode == BilevelMode
        asr_msg = "BilevelMode not implemented for multiple forecasting models"
        @assert length(model.forecast.networks) == 1 asr_msg
        return solve_bilevel(model, X, y, options.params)
    else
        # should never get here
        throw(ArgumentError("Invalid optimization method"))
    end
end

export Model,
    PredictiveModel,
    Plan,
    Assess,
    Policy,
    Forecast,
    set_forecast_model,
    forecast,
    compute_cost,
    train!
end
