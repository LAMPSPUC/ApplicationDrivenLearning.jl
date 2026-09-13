module ApplicationDrivenLearning

using Flux
using JuMP
using DiffOpt
import Tables
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

# normalizes the matrix / vector / Tables.jl inputs of the public entry points
include("data.jl")

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
    set_forecast_model(model::Model, forecast::PredictiveModel)

Attach a predictive (forecast) model to `model`.

The output size of `forecast` must match the number of [`Forecast`](@ref)
variables declared on `model`. If it has no `input_output_map`, a trivial one
mapping every input to every forecast variable is created. Its
`output_variables` are always reordered to follow `model.forecast_vars`, so that
the rows of a prediction line up with the forecast parameters of the plan model —
and any `output_names` follow their variables through that reordering.

The schema — `input_names`, `output_names` and `sample_key` — is declared on the
[`PredictiveModel`](@ref) and nowhere else:

```julia
set_forecast_model(
    model,
    PredictiveModel(
        Flux.Chain(Flux.Dense(2 => 1));
        input_names = [:temp, :hour],
    ),
)
```

This function used to accept a bare `Flux.Chain`/`Dense` and the same three
keywords. It no longer does, because the keywords and the `PredictiveModel`
fields of the same name were indexed by *different* orders — the keywords by
`model.forecast_vars`, the fields by the predictive model's own
`output_variables` — and supplying both silently picked one. There is now a
single place to say it.

Returns the stored [`PredictiveModel`](@ref).
"""
function set_forecast_model(model::Model, forecast::PredictiveModel)
    @assert forecast.output_size == length(model.forecast_vars) "Output size of forecast model must match number of forecast variables"

    input_output_map = forecast.input_output_map
    if isnothing(input_output_map)
        # no map: the single network reads the whole input and produces every
        # forecast variable
        input_output_map =
            [Dict(collect(1:forecast.input_size) => model.forecast_vars)]
    end

    # the stored names align with the predictive model's own `output_variables`,
    # so they have to follow those variables through the reordering below
    output_names =
        if isnothing(forecast.output_names) ||
           isnothing(forecast.output_variables)
            forecast.output_names
        else
            forecast.output_names[_find_elements_position(
                forecast.output_variables,
                model.forecast_vars,
            )]
        end

    # rebuild unconditionally: `output_variables` must follow
    # `model.forecast_vars` so that the rows of a prediction line up with the
    # plan model's forecast parameters
    return model.forecast = PredictiveModel(
        forecast.networks,
        input_output_map,
        model.forecast_vars,
        forecast.input_size,
        forecast.output_size;
        input_names = forecast.input_names,
        output_names = output_names,
        sample_key = forecast.sample_key,
    )
end

"""
    _build_plan_model_forecast_params(model::Model)

Turn the plan model's [`Forecast`](@ref) variables into `Parameter`
variables (initialised at zero) and record them in
`model.plan_forecast_params`. Their values are then set to the predictive
model output at every cost evaluation, and DiffOpt differentiates the plan
model with respect to them.
"""
function _build_plan_model_forecast_params(model::Model)
    # adds parametrized forecast variables using JuMP's `Parameter` set
    forecast_size = length(model.forecast_vars)
    # `copy` so that the two fields stay independent: `plan_forecast_vars`
    # returns the vector owned by `model`, not a fresh one
    model.plan_forecast_params = copy(plan_forecast_vars(model))
    return @constraint(
        model.plan,
        model.plan_forecast_params .∈ Parameter.(zeros(forecast_size))
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
include("optimizers/parallel.jl")
include("optimizers/parallel_mpi.jl")
include("optimizers/parallel_distributed.jl")
include("optimizers/gradient.jl")
include("optimizers/optim.jl")
include("optimizers/bilevel.jl")

"""
    _assert_forecast_model_set(model::Model)

Throw an `ArgumentError` unless a predictive model has been attached with
[`set_forecast_model`](@ref).

Called before the inputs are normalized, because deciding how to read the
realized values needs the forecast variables of the predictive model.
"""
function _assert_forecast_model_set(model::Model)
    if isnothing(model.forecast)
        throw(
            ArgumentError(
                "No forecast model set. Call set_forecast_model first.",
            ),
        )
    end
    return nothing
end

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
    absent = filter(!in(keys(data)), row_index)
    if !isempty(absent)
        throw(
            ArgumentError(
                "The realized values are missing the series of " *
                "$(length(absent)) forecast variable(s): " *
                "$(join([_forecast_base_name(f) for f in absent], ", ")).",
            ),
        )
    end
    n = size(data[row_index[1]], 1)
    tp = eltype(data[row_index[1]])
    Y = Matrix{tp}(undef, n, length(row_index))
    for (i, f) in enumerate(row_index)
        if length(data[f]) != n
            throw(
                ArgumentError(
                    "All forecast variable series must have the same length; " *
                    "`$(_forecast_base_name(f))` has $(length(data[f])) " *
                    "entry(ies) instead of $n.",
                ),
            )
        end
        Y[:, i] = data[f]
    end
    return Y
end

"""
    train!(model::Model, X, Y, options::Options)

Train the predictive model of `model` so that it minimizes the assessed cost
of the application.

...

# Arguments

  - `model::ApplicationDrivenLearning.Model`: model to train. Its forecast
    model must have been set with [`set_forecast_model`](@ref).
  - `X`: input data of size `(T, input_size)`. A matrix or vector, whose columns
    are taken in order, or a Tables.jl-compatible table such as a `DataFrame`,
    whose columns are selected by the predictive model's `input_names`.
  - `Y`: realized values of size `(T, output_size)`. A matrix or vector whose
    columns follow the predictive model output order, a Tables.jl-compatible
    table whose columns are matched to the [`Forecast`](@ref) variables by name,
    or a dictionary mapping each [`Forecast`](@ref) variable to its length-`T`
    series.

A table is always matched by name and never falls back to column order; see
[`set_forecast_model`](@ref) for how those names are declared.

  - `options::Options`: training mode and its parameters.

Returns a [`Solution`](@ref) with the best cost found and the corresponding
parameter vector. The predictive model of `model` is updated in place with
those parameters.
...
"""
function train!(
    model::Model,
    X::AbstractMatrix{<:Real},
    Y::AbstractMatrix{<:Real},
    options::Options,
)
    _assert_forecast_model_set(model)
    _assert_model_is_enough(_parallel_backend(options.params))

    # the parallel backends call `_compute_single_step_cost` directly instead of
    # going through `compute_cost`, so the parameters and the policy-fixing
    # constraint have to be in place before training starts
    _build(model)

    return _train!(options.mode, model, X, Y, options.params)
end

"""
    train!(build_model::Function, X, Y, options::Options)

Train a model built by `build_model`, a zero-argument function returning a
ready-to-solve [`Model`](@ref) — variables, constraints, objectives,
`set_optimizer` and [`set_forecast_model`](@ref) all done.

Equivalent to building the model and calling `train!` on it, except that the
function itself is available to the parallel backend. That is what
[`DistributedBackend`](@ref) needs: its workers never run your script, and a model
that has had `set_optimizer` called on it cannot be sent to them at all, so a
builder is the only way they can obtain one. Passing a `Model` under that backend
is an error.

Using one function for the whole run is also what makes the driver's model and the
workers' models the same problem by construction, rather than two descriptions
that have to be checked against each other.

!!! note

    The caller keeps no reference to the model this builds. The fitted parameters
    are in the returned `Solution`, so a fitted model is recovered with

    ```julia
    sol = train!(build_model, X, Y, options)
    m = build_model()
    ApplicationDrivenLearning.apply_params(m.forecast, sol.params)
    ```

    Both methods return a [`Solution`](@ref); a return type that depended on which
    one was called would be worse than this.
"""
function train!(
    build_model::Function,
    @nospecialize(X),
    @nospecialize(Y),
    options::Options,
)
    model = build_model()
    if !(model isa Model)
        throw(
            ArgumentError(
                "The model builder passed to `train!` must return an " *
                "`ApplicationDrivenLearning.Model`, got $(typeof(model)).",
            ),
        )
    end
    return train!(model, X, Y, _with_model_builder(options, build_model))
end

"""
    _train!(mode, model, X, Y, params)

Run the trainer registered for `mode` on already-normalized matrices.

`Options` already accepts any `Type{<:AbstractOptimizationMode}`
([`Options`](@ref)), so registering a new mode means declaring the singleton type
and adding one method below.
"""
function _train!(
    mode,
    ::Model,
    ::AbstractMatrix{<:Real},
    ::AbstractMatrix{<:Real},
    ::Dict{Symbol,Any},
)
    return throw(
        ArgumentError(
            "No trainer is registered for optimization mode `$mode`. Modes " *
            "provided by a package extension need that package loaded first: " *
            "`NLoptMode` requires `using NLopt`.",
        ),
    )
end

# train! with any other supported container: vectors, Tables.jl tables such as a
# `DataFrame`, and the `Dict{Forecast,Vector}` form, all normalized to matrices by
# `_to_matrices` before training starts.
#
# `X` and `Y` are `@nospecialize`d for the same reason as in `compute_cost`:
# specializing this wrapper on the container types pulls the whole training path
# through inference again, which measured in minutes rather than milliseconds.
function train!(
    model::Model,
    @nospecialize(X),
    @nospecialize(Y),
    options::Options,
)
    _assert_forecast_model_set(model)
    Xm, Ym = _to_matrices(X, Y, model.forecast)
    return train!(model, Xm, Ym, options)
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
