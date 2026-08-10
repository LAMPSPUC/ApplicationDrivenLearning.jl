using Optim

"""
    _train_with_nelder_mead!(model, X, Y, params)

Train the predictive model with the derivative-free Nelder-Mead algorithm from
Optim.jl, using the assessed cost as the objective.

See [`NelderMeadMode`](@ref) for the accepted `params`; any key other than
`initial_simplex` and `parameters` is forwarded to `Optim.Options`.
"""
function _train_with_nelder_mead!(
    model::Model,
    X::Matrix{<:Real},
    Y::Matrix{<:Real},
    params::Dict{Symbol,Any},
)

    # extract params
    initial_simplex = get(params, :initial_simplex, Optim.AffineSimplexer())
    parameters = get(params, :parameters, Optim.AdaptiveParameters())
    optim_params =
        filter(x -> !(x[1] in [:initial_simplex, :parameters]), params)
    optim_options = Optim.Options(; optim_params...)

    # fitness function
    function fitness(θ)
        apply_params(model.forecast, θ)
        return compute_cost(model, X, Y, false)
    end

    # call optimizer
    initial_sol = extract_params(model.forecast)
    res = Optim.optimize(
        fitness,
        initial_sol,
        NelderMead(;
            parameters = parameters,
            initial_simplex = initial_simplex,
        ),
        optim_options,
    )
    # update model parameters
    final_sol = Optim.minimizer(res)
    apply_params(model.forecast, final_sol)
    # return cost
    final_cost = Optim.minimum(res)
    return Solution(final_cost, final_sol)
end
