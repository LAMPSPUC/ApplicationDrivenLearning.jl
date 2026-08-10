module ApplicationDrivenLearning

using Flux
using JuMP
using DiffOpt
import ParametricOptInterface as POI
import Base.*, Base.+

# must come first: the files below expand `@timeit_debug` sections against the
# `_TIMER` declared here
include("timing.jl")

include("flux_utils.jl")

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
*(p::Policy, c::Number) = Policy(c * p.plan, c * p.assess)

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
*(p::Forecast, c::Number) = Forecast(c * p.plan, c * p.assess)

"""
    Base.getproperty(arr::AbstractArray{<:Policy}, sym::Symbol)

Allow accessing `.plan` and `.assess` on arrays of [`Policy`](@ref) variables.
Returns an array of the corresponding field values, preserving the shape and
axes of `arr` (so `x.plan[i]` always refers to the same element as `x[i].plan`).
Preserves all other properties by falling back to `getfield`.
"""
function Base.getproperty(arr::AbstractArray{<:Policy}, sym::Symbol)
    if sym === :plan
        return map(x -> x.plan, arr)
    elseif sym === :assess
        return map(x -> x.assess, arr)
    else
        # Fallback to original behavior for all other properties (e.g., .data, .axes for JuMP containers)
        return getfield(arr, sym)
    end
end

"""
    Base.getproperty(arr::AbstractArray{<:Forecast}, sym::Symbol)

Allow accessing `.plan` and `.assess` on arrays of [`Forecast`](@ref)
variables. Returns an array of the corresponding field values, preserving the
shape and axes of `arr` (so `d.plan[i]` always refers to the same element as
`d[i].plan`). Preserves all other properties by falling back to `getfield`.
"""
function Base.getproperty(arr::AbstractArray{<:Forecast}, sym::Symbol)
    if sym === :plan
        return map(x -> x.plan, arr)
    elseif sym === :assess
        return map(x -> x.assess, arr)
    else
        # Fallback to original behavior for all other properties (e.g., .data, .axes for JuMP containers)
        return getfield(arr, sym)
    end
end

include("predictive_model.jl")

"""
    ApplicationDrivenLearning._PolicyFixConstraint

Concrete type of the entries of the `assess_policy_fix` constraint vector, kept
as an alias so that the [`Model`](@ref) field holding them is concretely typed.
The constraints are built as `assess_policy_var == 0`, which JuMP always
represents as a `ScalarAffineFunction`-in-`EqualTo`.
"""
const _PolicyFixConstraint = JuMP.ConstraintRef{
    JuMP.Model,
    MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}},
    JuMP.ScalarShape,
}

"""
    Model <: JuMP.AbstractModel

Create an empty ApplicationDrivenLearning.Model with empty plan and assess
models, missing forecast model and default settings.

Besides the [`Policy`](@ref) and [`Forecast`](@ref) variable pairs, the model
keeps the plan-side and assess-side variables split into their own concretely
typed vectors. Those are what the per-sample cost evaluation actually iterates
over, so they are maintained as variables are declared rather than rebuilt on
every access.
"""
mutable struct Model <: JuMP.AbstractModel
    plan::JuMP.Model
    assess::JuMP.Model
    forecast::Union{PredictiveModel,Nothing}

    # variable arrays
    policy_vars::Vector{Policy{JuMP.VariableRef}}
    forecast_vars::Vector{Forecast{JuMP.VariableRef}}
    plan_forecast_params::Vector{JuMP.VariableRef}

    # plan/assess splits of the above, kept in sync by `JuMP.add_variable`
    _plan_policy_vars::Vector{JuMP.VariableRef}
    _assess_policy_vars::Vector{JuMP.VariableRef}
    _plan_forecast_vars::Vector{JuMP.VariableRef}
    _assess_forecast_vars::Vector{JuMP.VariableRef}

    # `assess_policy_fix` constraints, filled in by `_build`
    _assess_policy_fix::Vector{_PolicyFixConstraint}

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
            Vector{Policy{JuMP.VariableRef}}(),
            Vector{Forecast{JuMP.VariableRef}}(),
            Vector{JuMP.VariableRef}(),
            Vector{JuMP.VariableRef}(),
            Vector{JuMP.VariableRef}(),
            Vector{JuMP.VariableRef}(),
            Vector{JuMP.VariableRef}(),
            Vector{_PolicyFixConstraint}(),
            Dict{Symbol,Any}(),
            false,
        )
    end
end

"""
    plan_policy_vars(model::Model)

Return the vector of [`Policy`](@ref) variables that belong to the plan model,
in declaration order.

The vector is owned by `model` and is not a copy; do not mutate it.
"""
plan_policy_vars(model::Model) = model._plan_policy_vars

"""
    assess_policy_vars(model::Model)

Return the vector of [`Policy`](@ref) variables that belong to the assess
model, in declaration order (matching [`plan_policy_vars`](@ref)).

The vector is owned by `model` and is not a copy; do not mutate it.
"""
assess_policy_vars(model::Model) = model._assess_policy_vars

"""
    plan_forecast_vars(model::Model)

Return the vector of [`Forecast`](@ref) variables that belong to the plan
model, in declaration order.

The vector is owned by `model` and is not a copy; do not mutate it.
"""
plan_forecast_vars(model::Model) = model._plan_forecast_vars

"""
    assess_forecast_vars(model::Model)

Return the vector of [`Forecast`](@ref) variables that belong to the assess
model, in declaration order (matching [`plan_forecast_vars`](@ref)).

The vector is owned by `model` and is not a copy; do not mutate it.
"""
assess_forecast_vars(model::Model) = model._assess_forecast_vars

"""
    assess_policy_fix_cons(model::Model)

Return the `assess_policy_fix` constraints, which pin the assess
[`Policy`](@ref) variables to the values chosen by the plan model. Empty until
[`_build`](@ref) has been called.
"""
assess_policy_fix_cons(model::Model) = model._assess_policy_fix

"""
    set_forecast_model(model::Model, network)

Attach a predictive (forecast) model to `model`. `network` may be a
`Flux.Chain`, a `Flux.Dense` or an already built [`PredictiveModel`](@ref);
the two former are wrapped into a `PredictiveModel` automatically.

The output size of the predictive model must match the number of
[`Forecast`](@ref) variables declared on `model`. If the predictive model has
no `input_output_map`, a trivial one mapping every input to every forecast
variable is created. The stored model's `output_variables` are always
reordered to follow `model.forecast_vars`, so that the rows of a prediction
line up with the forecast parameters of the plan model.

Returns the stored [`PredictiveModel`](@ref).
"""
function set_forecast_model(
    model::Model,
    network::Union{PredictiveModel,Flux.Chain,Flux.Dense},
)
    if network isa PredictiveModel
        forecast = network
    else
        forecast = PredictiveModel(network)
    end
    @assert forecast.output_size == length(model.forecast_vars) "Output size of forecast model must match number of forecast variables"

    # set input_output_map of forecast model if not set
    if isnothing(forecast.input_output_map)
        forecast = PredictiveModel(
            deepcopy(forecast.networks),
            [Dict(collect(1:forecast.input_size) => model.forecast_vars)],
            model.forecast_vars,
            forecast.input_size,
            forecast.output_size,
        )
    end

    # make sure the same order apply on model.forecast_vars and model.forecast.output_variables
    if any(forecast.output_variables .!= model.forecast_vars)
        forecast = PredictiveModel(
            forecast.networks,
            forecast.input_output_map,
            model.forecast_vars,
            forecast.input_size,
            forecast.output_size,
        )
    end

    return model.forecast = forecast
end

"""
    _build_plan_model_forecast_params(model::Model)

Turn the plan model's [`Forecast`](@ref) variables into `MOI.Parameter`
variables (initialised at zero) and record them in
`model.plan_forecast_params`. Their values are then set to the predictive
model output at every cost evaluation, and DiffOpt differentiates the plan
model with respect to them.
"""
function _build_plan_model_forecast_params(model::Model)
    # adds parametrized forecast variables using MOI.Parameter
    forecast_size = length(model.forecast_vars)
    # `copy` so that the two fields stay independent: `plan_forecast_vars`
    # returns the vector owned by `model`, not a fresh one
    model.plan_forecast_params = copy(plan_forecast_vars(model))
    return @constraint(
        model.plan,
        model.plan_forecast_params .∈ MOI.Parameter.(zeros(forecast_size))
    )
end

"""
    _build_assess_model_policy_constraint(model::Model)

Add the `assess_policy_fix` constraint to the assess model, which pins each
assess [`Policy`](@ref) variable to the value chosen by the plan model. The
right-hand side is updated at every cost evaluation, and its dual is the
gradient of the assessed cost with respect to the policy.
"""
function _build_assess_model_policy_constraint(model::Model)
    cons = @constraint(
        model.assess,
        assess_policy_fix,
        assess_policy_vars(model) .== 0
    )
    # cached so that the per-sample loop does not go through the (untyped)
    # object dictionary of the assess model on every evaluation
    model._assess_policy_fix = cons
    return cons
end

"""
    _build(model::Model)

Add the variables and constraints required for cost computation to the plan
and assess models. Called automatically by [`compute_cost`](@ref); repeated
calls are no-ops.
"""
function _build(model::Model)
    if model.build
        return
    end
    model.build = true

    # build plan model
    _build_plan_model_forecast_params(model)
    return _build_assess_model_policy_constraint(model)
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
    _dict_to_var_indexed_matrix(data::Dict{<:Forecast,<:Vector}, row_index::Vector{<:Forecast})

Transform a dictionary that maps [`Forecast`](@ref) variables to their
realized series into a `(samples x variables)` matrix whose columns follow the
order of `row_index`.

Every variable in `row_index` must be a key of `data`, and all series must
have the same length.
"""
function _dict_to_var_indexed_matrix(
    data::Dict{<:Forecast,<:Vector},
    row_index::Vector{<:Forecast},
)
    n = size(data[row_index[1]], 1)
    tp = eltype(data[row_index[1]])
    Y = Matrix{tp}(undef, n, length(row_index))
    for (i, f) in enumerate(row_index)
        @assert length(data[f]) == n "All forecast variable series must have the same length"
        Y[:, i] = data[f]
    end
    return Y
end

"""
    train!(model::Model, X::Matrix{<:Real}, Y::Matrix{<:Real}, options::Options)
    train!(model::Model, X::Matrix{<:Real}, Y_dict::Dict{<:Forecast,<:Vector}, options::Options)

Train the predictive model of `model` so that it minimizes the assessed cost
of the application.

...

# Arguments

  - `model::ApplicationDrivenLearning.Model`: model to train. Its forecast
    model must have been set with [`set_forecast_model`](@ref).
  - `X::Matrix{<:Real}`: input data of size `(T, input_size)`.
  - `Y`: realized values, either a `(T, output_size)` matrix whose columns
    follow the predictive model output order, or a dictionary mapping each
    [`Forecast`](@ref) variable to its length-`T` series.
  - `options::Options`: training mode and its parameters.

Returns a [`Solution`](@ref) with the best cost found and the corresponding
parameter vector. The predictive model of `model` is updated in place with
those parameters.
...
"""
function train!(
    model::Model,
    X::Matrix{<:Real},
    Y::Matrix{<:Real},
    options::Options,
)
    if isnothing(model.forecast)
        throw(
            ArgumentError(
                "No forecast model set. Call set_forecast_model first.",
            ),
        )
    end

    # the MPI modes call `_compute_single_step_cost` directly instead of going
    # through `compute_cost`, so the parameters and the policy-fixing
    # constraint have to be in place before training starts
    _build(model)

    y = Y

    if options.mode == NelderMeadMode
        return _train_with_nelder_mead!(model, X, y, options.params)
    elseif options.mode == GradientMode
        return _train_with_gradient!(model, X, y, options.params)
    elseif options.mode == NelderMeadMPIMode
        return _train_with_nelder_mead_mpi!(model, X, y, options.params)
    elseif options.mode == GradientMPIMode
        return _train_with_gradient_mpi!(model, X, y, options.params)
    elseif options.mode == BilevelMode
        return _solve_bilevel(model, X, y, options.params)
    else
        # should never get here: Options rejects unknown modes on construction
        throw(ArgumentError("Invalid optimization method"))
    end
end

# train! with dictionary structured real data argument
function train!(
    model::Model,
    X::Matrix{<:Real},
    Y_dict::Dict{<:Forecast,<:Vector},
    options::Options,
)
    if isnothing(model.forecast)
        throw(
            ArgumentError(
                "No forecast model set. Call set_forecast_model first.",
            ),
        )
    end
    # transform dictionary data into ordered matrix
    return train!(
        model,
        X,
        _dict_to_var_indexed_matrix(Y_dict, model.forecast.output_variables),
        options,
    )
end

export Model,
    PredictiveModel,
    Plan,
    Assess,
    Policy,
    Forecast,
    set_forecast_model,
    compute_cost,
    train!
end
