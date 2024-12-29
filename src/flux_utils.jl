using Flux

function extract_flux_params(model::Union{Flux.Chain, Flux.Dense})
    θ = Flux.params(model)
    return reduce(vcat, [vec(p) for p in θ])
end

function fix_flux_params_single_model(model::Union{Flux.Chain, Flux.Dense}, θ::Vector{<:Real})
    i = 1
    for p in Flux.params(model)
        psize = prod(size(p))
        p .= reshape(θ[i:i+psize-1], size(p))
        i += psize
    end
    return model
end

function fix_flux_params_multi_model(
    models, 
    θ::Vector{<:Real}
)
    i = 1
    for model in models
        for p in Flux.params(model)
            psize = prod(size(p))
            p .= reshape(θ[i:i+psize-1], size(p))
            i += psize
        end
    end
    return models
end

function has_params(layer)
    try
        # Attempt to get parameters; if it works and isn't empty, return true
        return !isempty(Flux.params(layer))
    catch e
        # If there is an error (e.g., method not matching), assume no parameters
        return false
    end
end
