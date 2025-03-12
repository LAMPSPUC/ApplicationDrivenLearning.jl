using Flux
using Statistics
import Zygote
import Optimisers

"""
    PredictiveModel(networks, input_output_map, input_size, output_size)

Creates a predictive (forecast) model for the AppDrivenLearning module
from Flux models and input/output information.

...

# Arguments

  - `networks`: array of Flux models to be used.
  - `input_output_map::Vector{Dict{Vector{Int}, Vector{Int}}}`: array in the
    same ordering as networks of mappings from input indexes to output indexes
    on which the models should be applied.
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
    networks::Any
    input_output_map::Vector{Dict{Vector{Int},Vector{Int}}}
    input_size::Int
    output_size::Int

    function PredictiveModel(
        networks,
        input_output_map::Vector{Dict{Vector{Int},Vector{Int}}},
        input_size::Int,
        output_size::Int,
    )
        return new(
            deepcopy(networks),
            input_output_map,
            input_size,
            output_size,
        )
    end
end

"""
    PredictiveModel(networks::Flux.Chain)

When only one network is passed as a Chain object, input and output
indexes are directly extracted.
"""
function PredictiveModel(network::Flux.Chain)
    input_size = size(network[1].weight)[2]
    output_size = size(network[end].weight)[1]
    input_output_map = [Dict(collect(1:input_size) => collect(1:output_size))]
    return PredictiveModel(
        [deepcopy(network)],
        input_output_map,
        input_size,
        output_size,
    )
end

"""
    PredictiveModel(networks::Flux.Dense)

When only one network is passed as a Dense object, input and output
indexes are directly extracted.
"""
function PredictiveModel(network::Flux.Dense)
    input_size = size(network.weight)[2]
    output_size = size(network.weight)[1]
    input_output_map = [Dict(collect(1:input_size) => collect(1:output_size))]
    return PredictiveModel(
        [deepcopy(network)],
        input_output_map,
        input_size,
        output_size,
    )
end

"""
    PredictiveModel(networks::Flux.Chain, input_output_map::Dict{Vector{Int}, Vector{Int}})

When only one network is passed as a Chain object with explicit
input to output mapping, input and output sizes are directly extracted.
"""
function PredictiveModel(
    network::Flux.Chain,
    input_output_map::Dict{Vector{Int},Vector{Int}},
)
    param_layers = [layer for layer in network if has_params(layer)]
    network_input_size = size(param_layers[1].weight, 2)
    network_output_size = size(param_layers[end].weight, 1)
    for (input_idx, output_idx) in input_output_map
        @assert length(input_idx) == network_input_size "Input indexes length must match model input size."
        @assert length(output_idx) == network_output_size "Output indexes length must match model output size."
    end

    model_input_size = maximum(maximum.(keys(input_output_map)))
    model_output_size = maximum(maximum.(values(input_output_map)))
    return PredictiveModel(
        [deepcopy(network)],
        [input_output_map],
        model_input_size,
        model_output_size,
    )
end

"""
    PredictiveModel(networks::Flux.Dense, input_output_map::Dict{Vector{Int}, Vector{Int}})

When only one network is passed as a Dense object with explicit
input to output mapping, input and output sizes are directly extracted.
"""
function PredictiveModel(
    network::Flux.Dense,
    input_output_map::Dict{Vector{Int},Vector{Int}},
)
    network_input_size = size(network.weight)[2]
    network_output_size = size(network.weight)[1]
    for (input_idx, output_idx) in input_output_map
        @assert length(input_idx) == network_input_size "Input indexes length must match model input size."
        @assert length(output_idx) == network_output_size "Output indexes length must match model output size."
    end

    model_input_size = maximum(maximum.(keys(input_output_map)))
    model_output_size = maximum(maximum.(values(input_output_map)))
    return PredictiveModel(
        [deepcopy(network)],
        [input_output_map],
        model_input_size,
        model_output_size,
    )
end

function (model::PredictiveModel)(X::AbstractMatrix)
    pred_size = size(X, 2)
    n_networks = length(model.networks)
    Yhat = Zygote.Buffer(
        Matrix{eltype(X)}(undef, model.output_size, pred_size),
        (model.output_size, pred_size),
    )
    for inn = 1:n_networks
        io_map = model.input_output_map[inn]
        nn = model.networks[inn]
        for (input_idx, output_idx) in io_map
            Yhat[output_idx, :] = nn(X[input_idx, :])
        end
    end
    return copy(Yhat)
end

function (model::PredictiveModel)(x::AbstractVector)
    n_networks = length(model.networks)
    yhat = Zygote.Buffer(
        Vector{eltype(x)}(undef, model.output_size),
        model.output_size,
    )
    for inn = 1:n_networks
        io_map = model.input_output_map[inn]
        nn = model.networks[inn]
        for (input_idx, output_idx) in io_map
            yhat[output_idx] = nn(x[input_idx])
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
    dCdy::Vector{<:Real},
    X::Matrix{<:Real},
    opt_state,
)
    loss3(m, X) = mean(dCdy'm(X'))
    grad = Zygote.gradient(loss3, model, X)[1]
    return Optimisers.update!(opt_state, model, grad)
end
