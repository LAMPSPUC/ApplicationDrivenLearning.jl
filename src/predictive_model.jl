using Flux
using Statistics
import Zygote
import Functors
import Optimisers

"""
    _get_ordered_output_variables(input_output_map::Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}})

Get the ordered output variables from the input-output map.
"""
function _get_ordered_output_variables(
    input_output_map::Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
)
    return reduce(
        vcat,
        [reduce(vcat, values(iomap)) for iomap in input_output_map],
    )
end

"""
    _get_input_indices(input_output_map::Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}})

Get the input indices from the input-output map.
"""
function _get_input_indices(
    input_output_map::Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
)
    return unique(
        reduce(vcat, [reduce(vcat, keys(iomap)) for iomap in input_output_map]),
    )
end

"""
    _get_max_input_index(input_output_map::Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}})

Get the maximum input index from the input-output maps.
"""
function _get_max_input_index(
    input_output_map::Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
)
    return maximum(_get_input_indices(input_output_map))
end

"""
    _resolve_input_names(input_output_map, input_names)

Translate an `input_output_map` keyed by column *names* into the equivalent map
keyed by column *positions*, and return it together with the resolved
`input_names`.

Names are resolved once here, at construction, so that everything downstream —
the call operator, [`apply_gradient!`](@ref), the optimizers — keeps working on
the integer representation it always used. The names survive only as
`input_names`, which is what selects the columns of a table input.

When `input_names` is not given it is the set of names used by the map, in
alphabetical order. Pass `input_names` explicitly to choose the column order
instead; it must list every name the map refers to, and may list more, in which
case the extra columns are required in the input but left unused.
"""
function _resolve_input_names(
    input_output_map::Vector{<:Dict{Vector{Symbol},<:Vector{<:Forecast}}},
    input_names::Union{Vector{Symbol},Nothing},
)
    used = sort!(
        unique(
            reduce(
                vcat,
                [reduce(vcat, keys(iomap)) for iomap in input_output_map],
            ),
        ),
    )
    if isnothing(input_names)
        input_names = used
    else
        unknown = filter(!in(input_names), used)
        if !isempty(unknown)
            throw(
                ArgumentError(
                    "The `input_output_map` refers to the input column(s) " *
                    "$(join(string.(unknown), ", ")), which `input_names` does " *
                    "not list. `input_names` is $(join(string.(input_names), ", ")).",
                ),
            )
        end
    end
    position = Dict(name => i for (i, name) in enumerate(input_names))
    resolved = [
        Dict(
            [position[name] for name in key] => value for (key, value) in iomap
        ) for iomap in input_output_map
    ]
    return resolved, input_names
end

"""
    PredictiveModel(networks, input_output_map, output_variables, input_size, output_size; input_names, output_names)

Creates a predictive (forecast) model for the ApplicationDrivenLearning module
from Flux models and input/output information.

This is the fully explicit constructor; the convenience methods below derive
the missing arguments from the networks and the input/output map.

...

# Arguments

  - `networks`: array of Flux models to be used. The models are deep-copied,
    so the caller's objects are left untouched.
  - `input_output_map::Union{Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},Nothing}`: array in the
    same ordering as `networks` of mappings from input indexes to the
    [`Forecast`](@ref) variables that the corresponding model predicts. Use
    `nothing` to apply a single network directly to the whole input.
  - `output_variables::Union{Vector{<:Forecast},Nothing}`: forecast variables
    in the order in which the model produces them, i.e. the row order of a
    prediction.
  - `input_size::Int`: size of the input vector.
  - `output_size::Int`: size of the output vector.
  - `input_names::Union{Vector{Symbol},Nothing}`: name of each input column, in
    the order the model expects them. This is the model's declared input schema:
    it is what lets a table input be matched by name instead of by column order.
    `nothing` means the model declares no schema, and a table input is then
    rejected rather than guessed at.
  - `output_names::Union{Vector{Symbol},Nothing}`: name of the table column
    holding each variable of `output_variables`, in that same order. `nothing`
    means the declared names of the [`Forecast`](@ref) variables themselves are
    used.
  - `sample_key::Union{Symbol,Nothing}`: name of the column that identifies a
    *row*, the way `input_names` identifies a column — a timestamp or an id.
    When both `X` and `Y` are tables carrying it, the realized values are looked
    up by key for each row of `X` instead of being trusted to arrive in the same
    order. `nothing` means rows are matched by position.
    ...

# Example

Two `Forecast` variables predicted by one shared network from a different pair
of input columns each, plus a second network predicting two more variables:

```julia
model = ApplicationDrivenLearning.Model()
@variable(model, d[1:4], ApplicationDrivenLearning.Forecast)

pred_model = PredictiveModel(
    [Flux.Dense(2 => 1), Flux.Dense(1 => 2)],
    [Dict([1, 3] => [d[1]], [2, 3] => [d[2]]), Dict([4] => [d[3], d[4]])],
    [d[1], d[2], d[3], d[4]],
    4,
    4,
)
```
"""
struct PredictiveModel
    networks::AbstractVector
    input_output_map::Union{
        Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
        Nothing,
    }
    output_variables::Union{Vector{<:Forecast},Nothing}
    input_size::Int
    output_size::Int
    input_names::Union{Vector{Symbol},Nothing}
    output_names::Union{Vector{Symbol},Nothing}
    sample_key::Union{Symbol,Nothing}

    function PredictiveModel(
        networks::AbstractVector,
        input_output_map::Union{
            Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
            Nothing,
        },
        output_variables::Union{Vector{<:Forecast},Nothing},
        input_size::Int,
        output_size::Int;
        input_names::Union{Vector{Symbol},Nothing} = nothing,
        output_names::Union{Vector{Symbol},Nothing} = nothing,
        sample_key::Union{Symbol,Nothing} = nothing,
    )
        # every construction path funnels through here, so this is the one place
        # a declared schema has to be checked against the model it describes
        _check_names(input_names, input_size, "input_names", "takes", "input")
        _check_names(
            output_names,
            output_size,
            "output_names",
            "produces",
            "output",
        )
        return new(
            deepcopy(networks),
            input_output_map,
            output_variables,
            input_size,
            output_size,
            input_names,
            output_names,
            sample_key,
        )
    end
end

"""
    PredictiveModel(networks, input_output_map, output_variables, input_size, output_size, input_names, output_names, sample_key)

All-positional form of the constructor above.

Required, not merely convenient: `Functors.@functor` rebuilds the struct by
calling its constructor with every field in order, which is what happens on every
`Optimisers.update!` — so a field that is only reachable by keyword would make
the model untrainable.
"""
function PredictiveModel(
    networks::AbstractVector,
    input_output_map::Union{
        Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
        Nothing,
    },
    output_variables::Union{Vector{<:Forecast},Nothing},
    input_size::Int,
    output_size::Int,
    input_names::Union{Vector{Symbol},Nothing},
    output_names::Union{Vector{Symbol},Nothing},
    sample_key::Union{Symbol,Nothing},
)
    return PredictiveModel(
        networks,
        input_output_map,
        output_variables,
        input_size,
        output_size;
        input_names = input_names,
        output_names = output_names,
        sample_key = sample_key,
    )
end

"""
    _check_names(names, n, what, verb, noun)

Throw an `ArgumentError` unless `names` is `nothing` or a vector of `n` distinct
symbols. Duplicates matter as much as the count: two entries with the same name
would make one column feed two different slots.
"""
function _check_names(
    names::Union{Vector{Symbol},Nothing},
    n::Int,
    what::String,
    verb::String,
    noun::String,
)
    isnothing(names) && return nothing
    if length(names) != n
        throw(
            ArgumentError(
                "`$what` has $(length(names)) entry(ies) but the predictive " *
                "model $verb $n $noun(s): $(join(string.(names), ", ")).",
            ),
        )
    elseif !allunique(names)
        throw(
            ArgumentError(
                "`$what` must not repeat a name, got " *
                "$(join(string.(names), ", ")).",
            ),
        )
    end
    return nothing
end

"""
    PredictiveModel(networks::AbstractVector, input_output_map::Union{Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},Nothing})

Creates a predictive (forecast) model for the ApplicationDrivenLearning module
from a list of Flux models and an input/output map, one entry per model.

The list may mix model types (for example a `Flux.Dense` and a `Flux.Chain`);
`output_variables`, `input_size` and `output_size` are derived from the map.
"""
function PredictiveModel(
    networks::AbstractVector,
    input_output_map::Union{
        Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
        Nothing,
    };
    kwargs...,
)
    output_variables = _get_ordered_output_variables(input_output_map)
    input_size = _get_max_input_index(input_output_map)
    output_size = length(output_variables)
    return ApplicationDrivenLearning.PredictiveModel(
        networks,
        input_output_map,
        output_variables,
        input_size,
        output_size;
        kwargs...,
    )
end

"""
    PredictiveModel(networks::AbstractVector, input_output_map::Vector{<:Dict{Vector{Symbol},<:Vector{<:Forecast}}})

Creates a predictive (forecast) model whose `input_output_map` selects inputs by
column *name* rather than by column position, so that a table input cannot be
silently mispaired by having its columns in a different order.

```julia
pred_model = PredictiveModel(
    [Flux.Dense(2 => 1), Flux.Dense(1 => 1)],
    [Dict([:temp, :hour] => [d[1]]), Dict([:price] => [d[2]])],
)
```

The names become the model's `input_names`, in alphabetical order unless
`input_names` is passed explicitly to fix a different column order. Internally
the map is stored by position exactly as before, so this is purely a change of
how the map is written and how table columns are selected.
"""
function PredictiveModel(
    networks::AbstractVector,
    input_output_map::Vector{<:Dict{Vector{Symbol},<:Vector{<:Forecast}}};
    input_names::Union{Vector{Symbol},Nothing} = nothing,
    kwargs...,
)
    resolved, names = _resolve_input_names(input_output_map, input_names)
    output_variables = _get_ordered_output_variables(resolved)
    return ApplicationDrivenLearning.PredictiveModel(
        networks,
        resolved,
        output_variables,
        length(names),
        length(output_variables);
        input_names = names,
        kwargs...,
    )
end

"""
    _network_io_sizes(network)

Input and output size of a single Flux network, read off its parameterised
layers.
"""
function _network_io_sizes(network::Flux.Chain)
    param_layers = [layer for layer in network if _has_params(layer)]
    return size(param_layers[1].weight, 2), size(param_layers[end].weight, 1)
end

_network_io_sizes(network::Flux.Dense) = reverse(size(network.weight))

"""
    PredictiveModel(network::Union{Flux.Chain,Flux.Dense})

When only one network is passed, its input and output sizes are taken from its
parameterised layers and the `input_output_map` is set to `nothing`, meaning the
network is applied directly to the whole input.

Such a model has no map to carry column names, so pass `input_names` (and
`output_names`, if the [`Forecast`](@ref) variables are not named after the
columns) to be able to hand it a table:

```julia
set_forecast_model(
    model,
    Flux.Chain(Flux.Dense(2 => 1));
    input_names = [:temp, :hour],
)
```
"""
function PredictiveModel(network::Union{Flux.Chain,Flux.Dense}; kwargs...)
    input_size, output_size = _network_io_sizes(network)
    return PredictiveModel(
        [network],
        nothing,
        nothing,
        input_size,
        output_size;
        kwargs...,
    )
end

"""
    PredictiveModel(network::Union{Flux.Chain,Flux.Dense}, input_output_map::Dict)

When only one network is passed with an explicit input to output mapping, the
input and output sizes are derived from the map. Each entry must have as many
input indexes — or input names — as the network's input size, and as many
forecast variables as its output size.
"""
function PredictiveModel(
    network::Union{Flux.Chain,Flux.Dense},
    input_output_map::Union{
        Dict{Vector{Int},<:Vector{<:Forecast}},
        Dict{Vector{Symbol},<:Vector{<:Forecast}},
    };
    kwargs...,
)
    network_input_size, network_output_size = _network_io_sizes(network)
    for (input_idx, output_idx) in input_output_map
        @assert length(input_idx) == network_input_size "Input indexes length must match model input size."
        @assert length(output_idx) == network_output_size "Output indexes length must match model output size."
    end
    return PredictiveModel([network], [input_output_map]; kwargs...)
end

"""
    Flux.trainable(model::PredictiveModel)

Make PredictiveModel compatible with Flux's training interface by
specifying that only the networks field is trainable.
"""
Flux.trainable(model::PredictiveModel) = (networks = model.networks,)

# Tells Flux to only look at the 'network' field when setting up or traversing
Functors.@functor PredictiveModel (networks,)

"""
    _find_elements_position(vec, elements)

Return the position in `vec` of each entry of `elements`. Used to map the
forecast variables produced by a network onto the rows of the prediction
matrix. Entries not present in `vec` yield `nothing`.
"""
function _find_elements_position(vec, elements)
    return [findfirst(i -> i == j, vec) for j in elements]
end

"""
    (model::PredictiveModel)(X::AbstractMatrix)

Predict the output of the model for a given input matrix.
"""
function (model::PredictiveModel)(X::AbstractMatrix)
    pred_size = size(X, 2)  # length of the input data
    n_networks = length(model.networks)  # number of networks in the model
    # buffer to store the predicted output
    Yhat = Zygote.Buffer(
        Matrix{eltype(X)}(undef, model.output_size, pred_size),
        (model.output_size, pred_size),
    )

    # no input-output map case
    if isnothing(model.input_output_map)
        # there should only be one network in the model
        @assert n_networks == 1 "There should only be one network in the predictive model when there is no input-output map."
        # apply the network to the input
        return model.networks[1](X)
    end

    for inn = 1:n_networks
        nn = model.networks[inn]
        for (input_idx, output_idx) in model.input_output_map[inn]
            Yhat[
                _find_elements_position(model.output_variables, output_idx),
                :,
            ] = nn(X[input_idx, :])
        end
    end
    return copy(Yhat)
end

"""
    (model::PredictiveModel)(x::AbstractVector)

Predict the output of the model for a given input vector.
If the model has no input-output map, the network is applied directly to the input.
"""
function (model::PredictiveModel)(x::AbstractVector)
    n_networks = length(model.networks)  # number of networks in the model
    # buffer to store the predicted output
    yhat = Zygote.Buffer(
        Vector{eltype(x)}(undef, model.output_size),
        model.output_size,
    )

    # no input-output map case
    if isnothing(model.input_output_map)
        # there should only be one network in the model
        @assert n_networks == 1 "There should only be one network in the predictive model when there is no input-output map."
        # apply the network to the input
        return model.networks[1](x)
    end

    # input-output map case
    for inn = 1:n_networks
        # gets the input-output map for the current network
        nn = model.networks[inn]
        for (input_idx, output_idx) in model.input_output_map[inn]
            # set the predicted output for the current output variables indices
            out_y_idx =
                _find_elements_position(model.output_variables, output_idx)
            yhat[out_y_idx] = nn(x[input_idx])
        end
    end
    return copy(yhat)
end

"""
    extract_params(model)

Extract the parameters of a PredictiveModel into a single vector.
"""
function extract_params(model::PredictiveModel)
    @timeit_debug _TIMER "extract_params" begin
        # NOTE: keep the splat. `reduce(vcat, xs)` returns `xs[1]` untouched
        # when there is a single network, and `_extract_flux_params` itself
        # returns `vec(p)` - an alias of the live weights - when that network
        # has a single trainable array. `vcat` always copies, which is what
        # callers such as `best_θ` in the gradient loop rely on.
        return vcat([_extract_flux_params(nn) for nn in model.networks]...)
    end
end

"""
    apply_params(model, θ)

Return model after fixing the parameters from an adequate vector of parameters.
"""
function apply_params(model::PredictiveModel, θ)
    @timeit_debug _TIMER "apply_params" begin
        return _fix_flux_params_multi_model(model.networks, θ)
    end
end

"""
    apply_gradient!(model, dCdy, X, opt_state)

Apply per-sample cost gradients to the model parameters.

The optimization layer provides `dCdy`, the gradient of the assessment cost with
respect to the forecasts, for each sample. Because the optimization step itself
is not differentiable by the AD backend, these gradients are propagated through
the forecast model with a linear surrogate loss whose parameter-gradient equals
the chain-rule term `(1/T) Σₜ (dC/dŷₜ)·(dŷₜ/dθ)`.

...

# Arguments

  - `model::PredictiveModel`: model to be updated.
  - `dCdy::AbstractMatrix{<:Real}`: per-sample cost gradients, size
    `(T, output_size)`, with row `t` aligned to sample `t` (row `t` of `X`).
  - `X::AbstractMatrix{<:Real}`: input data, size `(T, input_size)`.
  - `opt_state`: Optimisers optimisation state.
    ...
"""
function apply_gradient!(
    model::PredictiveModel,
    dCdy::AbstractMatrix{<:Real},
    X::AbstractMatrix{<:Real},
    opt_state,
)
    surrogate_loss(m, X) = sum(dCdy' .* m(X')) / size(X, 1)
    grad = @timeit_debug _TIMER "zygote_backward" Zygote.gradient(
        surrogate_loss,
        model,
        X,
    )[1]
    return @timeit_debug _TIMER "optimiser_update" Optimisers.update!(
        opt_state,
        model,
        grad,
    )
end
