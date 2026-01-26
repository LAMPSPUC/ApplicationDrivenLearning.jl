using Optim

function train_with_nelder_mead!(
    model::Model,
    X::Matrix{<:Real},
    Y::Dict{<:Forecast, <:Vector},
    params::Dict{Symbol,Any},
)

    # extract params
    initial_simplex = get(params, :initial_simplex, Optim.AffineSimplexer())
    parameters = get(params, :parameters, Optim.AdaptiveParameters())
    filter!(x -> !(x[1] in [:initial_simplex, :parameters]), params)
    optim_options = Optim.Options(; params...)

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
    final_cost = minimum(res)
    return Solution(final_cost, final_sol)
end
