using Flux
using Statistics
import Zygote
import Optimisers

include("variable_indexed_structs.jl")

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
    networks::Union{Vector{<:Flux.Chain},Vector{<:Flux.Dense}}
    input_output_map::Union{Vector{Dict{Vector{Int},Vector{Forecast{JuMP.VariableRef}}}},Nothing}
    input_size::Int
    output_size::Int

    function PredictiveModel(
        networks::Union{Vector{<:Flux.Chain},Vector{<:Flux.Dense}},
        input_output_map::Union{Vector{Dict{Vector{Int},Vector{Forecast{JuMP.VariableRef}}}},Nothing},
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
indexes are directly extracted and the input_output_map is set to nothing.
"""
function PredictiveModel(network::Flux.Chain)
    param_layers = [layer for layer in network if has_params(layer)]
    input_size = size(param_layers[1].weight, 2)
    output_size = size(param_layers[end].weight, 1)
    input_output_map = nothing
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
indexes are directly extracted and the input_output_map is set to nothing.
"""
function PredictiveModel(network::Flux.Dense)
    input_size = size(network.weight)[2]
    output_size = size(network.weight)[1]
    input_output_map = nothing
    return PredictiveModel(
        [deepcopy(network)],
        input_output_map,
        input_size,
        output_size,
    )
end

"""
    PredictiveModel(networks::Flux.Chain, input_output_map::Dict{Vector{Int}, Vector{Forecast}})

When only one network is passed as a Chain object with explicit
input to output mapping, input and output sizes are directly extracted.
"""
function PredictiveModel(
    network::Flux.Chain,
    input_output_map::Dict{Vector{Int},Vector{Forecast}},
)
    param_layers = [layer for layer in network if has_params(layer)]
    network_input_size = size(param_layers[1].weight, 2)
    network_output_size = size(param_layers[end].weight, 1)
    for (input_idx, output_idx) in input_output_map
        @assert length(input_idx) == network_input_size "Input indexes length must match model input size."
        @assert length(output_idx) == network_output_size "Output indexes length must match model output size."
    end

    model_input_size = maximum(maximum.(keys(input_output_map)))
    model_output_size = length(
        unique(
            reduce(vcat, values(input_output_map))
        )
    )
    return PredictiveModel(
        [deepcopy(network)],
        [input_output_map],
        model_input_size,
        model_output_size,
    )
end

"""
    PredictiveModel(networks::Flux.Dense, input_output_map::Dict{Vector{Int}, Vector{Forecast}})

When only one network is passed as a Dense object with explicit
input to output mapping, input and output sizes are directly extracted.
"""
function PredictiveModel(
    network::Flux.Dense,
    input_output_map::Dict{Vector{Int},Vector{Forecast}},
)
    network_input_size = size(network.weight)[2]
    network_output_size = size(network.weight)[1]
    for (input_idx, output_idx) in input_output_map
        @assert length(input_idx) == network_input_size "Input indexes length must match model input size."
        @assert length(output_idx) == network_output_size "Output indexes length must match model output size."
    end

    model_input_size = maximum(maximum.(keys(input_output_map)))
    model_output_size = length(
        unique(
            reduce(vcat, values(input_output_map))
        )
    )
    return PredictiveModel(
        [deepcopy(network)],
        [input_output_map],
        model_input_size,
        model_output_size,
    )
end

"""
    output_variables(model::PredictiveModel)

Return the variables that are output by the model.
"""
function output_variables(model::PredictiveModel)
    return unique(
        reduce(
            vcat, 
            [
                reduce(
                    vcat,
                    values(iomap)
                )
                for iomap in model.input_output_map
            ]
        )
    )
end

"""
    (model::PredictiveModel)(X::AbstractMatrix)

Predict the output of the model for a given input matrix.
"""
function (model::PredictiveModel)(X::AbstractMatrix)
    pred_size = size(X, 2)
    n_networks = length(model.networks)
    Yhat = Zygote.Buffer(
        Matrix{eltype(X)}(undef, model.output_size, pred_size),
        (model.output_size, pred_size),
    )
    i = 1
    out_idx = Vector{Forecast{JuMP.VariableRef}}(undef, model.output_size)
    for inn = 1:n_networks
        io_map = model.input_output_map[inn]
        nn = model.networks[inn]
        for (input_idx, output_idx) in io_map
            nn_input_pred = nn(X[input_idx, :])
            Yhat[i:i+length(output_idx)-1, :] = nn_input_pred
            out_idx[i:i+length(output_idx)-1] = output_idx
            i += length(output_idx)
        end
    end
    return VariableIndexedMatrix(copy(Yhat), out_idx)
end

"""
    (model::PredictiveModel)(x::AbstractVector)

Predict the output of the model for a given input vector.
"""
function (model::PredictiveModel)(x::AbstractVector)
    n_networks = length(model.networks)
    yhat = Zygote.Buffer(
        Vector{eltype(x)}(undef, model.output_size),
        model.output_size,
    )
    i = 1
    out_idx = Vector{Forecast{JuMP.VariableRef}}(undef, model.output_size)
    for inn = 1:n_networks
        io_map = model.input_output_map[inn]
        nn = model.networks[inn]
        for (input_idx, output_idx) in io_map
            nn_input_pred = nn(x[input_idx])
            yhat[i:i+length(output_idx)-1] = nn_input_pred
            out_idx[i:i+length(output_idx)-1] = output_idx
            i += length(output_idx)
        end
    end
    return VariableIndexedVector(copy(yhat), out_idx)
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
