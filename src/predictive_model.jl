using Flux
using Statistics
import Zygote
import Functors
import Optimisers

"""
    get_ordered_output_variables(input_output_map::Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}})

Get the ordered output variables from the input-output map.
"""
function get_ordered_output_variables(
    input_output_map::Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
)
    return reduce(
        vcat,
        [reduce(vcat, values(iomap)) for iomap in input_output_map],
    )
end

"""
    get_input_indices(input_output_map::Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}})

Get the input indices from the input-output map.
"""
function get_input_indices(
    input_output_map::Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
)
    return unique(
        reduce(vcat, [reduce(vcat, keys(iomap)) for iomap in input_output_map]),
    )
end

"""
    get_max_input_index(input_output_map::Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}})

Get the maximum input index from the input-output maps.
"""
function get_max_input_index(
    input_output_map::Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
)
    return maximum(get_input_indices(input_output_map))
end

"""
    PredictiveModel(networks, input_output_map, input_size, output_size)

Creates a predictive (forecast) model for the AppDrivenLearning module
from Flux models and input/output information.

...

# Arguments

  - `networks`: array of Flux models to be used.
  - `input_output_map::Union{Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},Nothing}`: array in the
    same ordering as networks of mappings from input indexes to output indexes
    on which the models should be applied.
  - `output_variables::Union{Vector{<:Forecast},Nothing}`: array of output variables to be used.
  - `input_size::Int`: size of the input vector.
  - `output_size::Int`: size of the output vector.
    ...

# Example

```
julia> pred_model = PredictiveModel(
        [Flux.Dense(1 => 1), Flux.Dense(3 => 2)],
        [
            Dict([1] => [1], [1] => [2]),
            Dict([1,2,3] => [3,4], [1,4,5] => [5,6])
        ],
        5,
        6
    );
```
"""
struct PredictiveModel
    networks::Union{Vector{<:Flux.Chain},Vector{<:Flux.Dense}}
    input_output_map::Union{
        Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
        Nothing,
    }
    output_variables::Union{Vector{<:Forecast},Nothing}
    input_size::Int
    output_size::Int

    function PredictiveModel(
        networks::Union{Vector{<:Flux.Chain},Vector{<:Flux.Dense}},
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
    PredictiveModel(networks::Union{Vector{<:Flux.Chain},Vector{<:Flux.Dense}}, input_output_map::Union{Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},Nothing})

Creates a predictive (forecast) model for the ApplicationDrivenLearning module
from Flux models and input/output map.
"""
function PredictiveModel(
    networks::Union{Vector{<:Flux.Chain},Vector{<:Flux.Dense}},
    input_output_map::Union{
        Vector{<:Dict{Vector{Int},<:Vector{<:Forecast}}},
        Nothing,
    },
)
    output_variables = get_ordered_output_variables(input_output_map)
    input_size = get_max_input_index(input_output_map)
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
    PredictiveModel(networks::Flux.Chain)

When only one network is passed as a Chain object, input and output
indexes are directly extracted and the input_output_map is set to nothing.
"""
function PredictiveModel(network::Flux.Chain)
    param_layers = [layer for layer in network if has_params(layer)]
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
    PredictiveModel(networks::Flux.Dense)

When only one network is passed as a Dense object, input and output
indexes are directly extracted and the input_output_map is set to nothing.
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
    PredictiveModel(networks::Flux.Chain, input_output_map::Dict{Vector{Int}, <:Vector{<:Forecast}})

When only one network is passed as a Chain object with explicit
input to output mapping, input and output sizes are directly extracted.
"""
function PredictiveModel(
    network::Flux.Chain,
    input_output_map::Dict{Vector{Int},<:Vector{<:Forecast}},
)
    param_layers = [layer for layer in network if has_params(layer)]
    network_input_size = size(param_layers[1].weight, 2)
    network_output_size = size(param_layers[end].weight, 1)
    for (input_idx, output_idx) in input_output_map
        @assert length(input_idx) == network_input_size "Input indexes length must match model input size."
        @assert length(output_idx) == network_output_size "Output indexes length must match model output size."
    end

    output_variables = get_ordered_output_variables([input_output_map])
    input_size = get_max_input_index([input_output_map])
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
    PredictiveModel(networks::Flux.Dense, input_output_map::Dict{Vector{Int}, <:Vector{<:Forecast}})

When only one network is passed as a Dense object with explicit
input to output mapping, input and output sizes are directly extracted.
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

    output_variables = get_ordered_output_variables([input_output_map])
    input_size = get_max_input_index([input_output_map])
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

# helper function
function find_elements_position(vec, elements)
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
    if model.input_output_map == nothing
        # there should only be one network in the model
        @assert n_networks == 1 "There should only be one network in the predictive model when there is no input-output map."
        # apply the network to the input
        return model.networks[1](X)
    end

    for inn = 1:n_networks
        nn = model.networks[inn]
        for (input_idx, output_idx) in model.input_output_map[inn]
            Yhat[find_elements_position(model.output_variables, output_idx), :] = nn(X[input_idx, :])
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
    if model.input_output_map == nothing
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
            out_y_idx = find_elements_position(model.output_variables, output_idx)
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
    return vcat([extract_flux_params(nn) for nn in model.networks]...)
end

"""
    apply_params(model, θ)

Return model after fixing the parameters from an adequate vector of parameters.
"""
function apply_params(model::PredictiveModel, θ)
    return fix_flux_params_multi_model(model.networks, θ)
end

"""
    apply_gradient!(model, dCdy, X, rule)

Apply a gradient vector to the model parameters.

...

# Arguments

  - `model::PredictiveModel`: model to be updated.
  - `dCdy::Vector{<:Real}`: gradient vector.
  - `X::Matrix{<:Real}`: input data.
  - `rule`: Optimisation rule.
    ...
"""
function apply_gradient!(
    model::PredictiveModel,
    dCdy::AbstractVector{<:Real},
    X::Matrix{<:Real},
    opt_state,
)
    loss3(m, X) = mean(dCdy'm(X'))
    grad = Zygote.gradient(loss3, model, X)[1]
    return Optimisers.update!(opt_state, model, grad)
end
