"""
    Solution{C<:Real,P<:Real}

Result of a [`train!`](@ref) call.

The cost and the parameters are stored with their own concrete element types,
which usually differ: costs come from the solver and are `Float64`, while
parameters follow the precision of the Flux model (often `Float32`).

...

# Fields

  - `cost::C`: best assessed cost found during training, averaged over the
    training samples.
  - `params::Vector{P}`: predictive model parameters that achieve `cost`,
    flattened in the order used by
    [`extract_params`](@ref ApplicationDrivenLearning.extract_params). The
    model itself is also updated in place with these values.
    ...
"""
struct Solution{C<:Real,P<:Real}
    cost::C
    params::Vector{P}
end
