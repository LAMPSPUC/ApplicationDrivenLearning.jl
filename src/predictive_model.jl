using Flux, Zygote

"""
Predictive Model

Abstracts the predictive model for the AppDrivenLearning module using Flux.
"""
struct PredictiveModel
    network
    input_size::Int
    output_size::Int
    
    function PredictiveModel(
        network,
        input_size::Int,
        output_size::Int
    )
        return new(deepcopy(network), input_size, output_size)
    end

    function PredictiveModel(
        network::Flux.Chain
    )
        input_size = size(network[1].weight)[2]
        output_size = size(network[end].weight)[1]
        return new(deepcopy(network), input_size, output_size)
    end

    function PredictiveModel(
        network::Flux.Dense
    )
        input_size = size(network.weight)[2]
        output_size = size(network.weight)[1]
        return new(deepcopy(network), input_size, output_size)
    end
end

(model::PredictiveModel)(X::AbstractMatrix) = model.network(X)
(model::PredictiveModel)(x::AbstractVector) = model.network(x)

function extract_params(model::PredictiveModel)
    return extract_flux_params(model.network)
end

function apply_params(model::PredictiveModel, θ)
    return fix_flux_params(model.network, θ)
end

function get_model_weights(model::PredictiveModel)
    return Flux.params(model.network)
end

function apply_gradient!(
    model::PredictiveModel, 
    dCdy::Vector{<:Real},
    optimizer
)
    ps = get_model_weights(model)
    loss(x, y) = dCdy'model(x)
    train_data = [(ones(Float32, model.input_size), zeros(Float32, model.output_size))]
    Flux.train!(loss, ps, train_data, optimizer)
end