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
    PredictiveModel(networks, input_output_map, output_variables, input_size, output_size)

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

    function PredictiveModel(
        networks::AbstractVector,
        input_output_map::Union{
            Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
            Nothing,
        },
        output_variables::Union{Vector{<:Forecast},Nothing},
        input_size::Int,
        output_size::Int,
    )
        return new(
            deepcopy(networks),
            input_output_map,
            output_variables,
            input_size,
            output_size,
        )
    end
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
    },
)
    output_variables = _get_ordered_output_variables(input_output_map)
    input_size = _get_max_input_index(input_output_map)
    output_size = length(output_variables)
    return ApplicationDrivenLearning.PredictiveModel(
        networks,
        input_output_map,
        output_variables,
        input_size,
        output_size,
    )
end

"""
    PredictiveModel(network::Flux.Chain)

When only one network is passed as a `Flux.Chain` object, the input and output
sizes are taken from its first and last parameterised layers and the
`input_output_map` is set to `nothing`, meaning the chain is applied directly
to the whole input.
"""
function PredictiveModel(network::Flux.Chain)
    param_layers = [layer for layer in network if _has_params(layer)]
    input_size = size(param_layers[1].weight, 2)
    output_size = size(param_layers[end].weight, 1)
    return PredictiveModel(
        [deepcopy(network)],
        nothing,
        nothing,
        input_size,
        output_size,
    )
end

"""
    PredictiveModel(network::Flux.Dense)

When only one network is passed as a `Flux.Dense` object, the input and output
sizes are taken from its weight matrix and the `input_output_map` is set to
`nothing`, meaning the layer is applied directly to the whole input.
"""
function PredictiveModel(network::Flux.Dense)
    input_size = size(network.weight)[2]
    output_size = size(network.weight)[1]
    return PredictiveModel(
        [deepcopy(network)],
        nothing,
        nothing,
        input_size,
        output_size,
    )
end

"""
    PredictiveModel(network::Flux.Chain, input_output_map::Dict{Vector{Int}, <:Vector{<:Forecast}})

When only one network is passed as a `Flux.Chain` object with an explicit
input to output mapping, the input and output sizes are derived from the map.
Each entry must have as many input indexes as the chain's input size and as
many forecast variables as its output size.
"""
function PredictiveModel(
    network::Flux.Chain,
    input_output_map::Dict{Vector{Int},<:Vector{<:Forecast}},
)
    param_layers = [layer for layer in network if _has_params(layer)]
    network_input_size = size(param_layers[1].weight, 2)
    network_output_size = size(param_layers[end].weight, 1)
    for (input_idx, output_idx) in input_output_map
        @assert length(input_idx) == network_input_size "Input indexes length must match model input size."
        @assert length(output_idx) == network_output_size "Output indexes length must match model output size."
    end

    output_variables = _get_ordered_output_variables([input_output_map])
    input_size = _get_max_input_index([input_output_map])
    output_size = length(output_variables)
    return PredictiveModel(
        [deepcopy(network)],
        [input_output_map],
        output_variables,
        input_size,
        output_size,
    )
end

"""
    PredictiveModel(network::Flux.Dense, input_output_map::Dict{Vector{Int}, <:Vector{<:Forecast}})

When only one network is passed as a `Flux.Dense` object with an explicit
input to output mapping, the input and output sizes are derived from the map.
Each entry must have as many input indexes as the layer's input size and as
many forecast variables as its output size.
"""
function PredictiveModel(
    network::Flux.Dense,
    input_output_map::Dict{Vector{Int},<:Vector{<:Forecast}},
)
    network_input_size = size(network.weight)[2]
    network_output_size = size(network.weight)[1]
    for (input_idx, output_idx) in input_output_map
        @assert length(input_idx) == network_input_size "Input indexes length must match model input size."
        @assert length(output_idx) == network_output_size "Output indexes length must match model output size."
    end

    output_variables = _get_ordered_output_variables([input_output_map])
    input_size = _get_max_input_index([input_output_map])
    output_size = length(output_variables)
    return PredictiveModel(
        [deepcopy(network)],
        [input_output_map],
        output_variables,
        input_size,
        output_size,
    )
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
    @timeit_debug TIMER "extract_params" begin
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
    @timeit_debug TIMER "apply_params" begin
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
  - `X::Matrix{<:Real}`: input data, size `(T, input_size)`.
  - `opt_state`: Optimisers optimisation state.
    ...
"""
function apply_gradient!(
    model::PredictiveModel,
    dCdy::AbstractMatrix{<:Real},
    X::Matrix{<:Real},
    opt_state,
)
    surrogate_loss(m, X) = sum(dCdy' .* m(X')) / size(X, 1)
    grad = @timeit_debug TIMER "zygote_backward" Zygote.gradient(
        surrogate_loss,
        model,
        X,
    )[1]
    return @timeit_debug TIMER "optimiser_update" Optimisers.update!(
        opt_state,
        model,
        grad,
    )
end
