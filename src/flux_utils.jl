using Flux

"""
    extract_flux_params(model)

Extract the parameters of a Flux model (Flux.Chain or Flux.Dense) into a single
vector.
"""
function extract_flux_params(model::Union{Flux.Chain,Flux.Dense})
    θ = Flux.params(model)
    return reduce(vcat, [vec(p) for p in θ])
end

"""
    fix_flux_params_single_model(model, θ)

Return model after fixing the parameters from an adequate vector of parameters.
"""
function fix_flux_params_single_model(
    model::Union{Flux.Chain,Flux.Dense},
    θ::Vector{<:Real},
)
    i = 1
    for p in Flux.params(model)
        psize = prod(size(p))
        p .= reshape(θ[i:i+psize-1], size(p))
        i += psize
    end
    return model
end

"""
    fix_flux_params_multi_model(models, θ)

Return iterable of models after fixing the parameters from an adequate vector
of parameters.
"""
function fix_flux_params_multi_model(models, θ::Vector{<:Real})
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

"""
    has_params(layer)

Check if a Flux layer has parameters.
"""
function has_params(layer)
    try
        # Attempt to get parameters; if it works and isn't empty, return true
        return !isempty(Flux.params(layer))
    catch e
        # If there is an error (e.g. method not matching), assume no parameters
        return false
    end
end
