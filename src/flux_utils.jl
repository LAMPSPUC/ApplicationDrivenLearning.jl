using Flux

"""
    _extract_flux_params(model)

Extract the trainable parameters of any Flux model or layer into a single
flat vector, in `Flux.trainables` order.
"""
function _extract_flux_params(model)
    θ = Flux.trainables(model)
    return reduce(vcat, [vec(p) for p in θ])
end

"""
    _fix_flux_params_single_model(model, θ)

Return the model after overwriting its trainable parameters in place with the
values of `θ`, which must be laid out as produced by
`_extract_flux_params`.
"""
function _fix_flux_params_single_model(model, θ::Vector{<:Real})
    i = 1
    for p in Flux.trainables(model)
        psize = prod(size(p))
        p .= reshape(θ[i:i+psize-1], size(p))
        i += psize
    end
    return model
end

"""
    _fix_flux_params_multi_model(models, θ)

Return the iterable of models after overwriting their trainable parameters in
place with the values of `θ`, concatenated in model order as produced by
[`extract_params`](@ref ApplicationDrivenLearning.extract_params).
"""
function _fix_flux_params_multi_model(models, θ::Vector{<:Real})
    i = 1
    for model in models
        for p in Flux.trainables(model)
            psize = prod(size(p))
            p .= reshape(θ[i:i+psize-1], size(p))
            i += psize
        end
    end
    return models
end

"""
    _has_params(layer)

Check if a Flux layer has parameters.
"""
function _has_params(layer)
    try
        # Attempt to get trainable parameters; if it works and isn't empty, return true
        return !isempty(Flux.trainable(layer))
    catch e
        # If there is an error (e.g. method not matching), assume no parameters
        return false
    end
end
