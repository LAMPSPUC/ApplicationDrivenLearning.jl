using Optim

function train_with_nelder_mead!(
    model::Model,
    X::Matrix{<:Real},
    Y::Matrix{<:Real},
    params::Dict{Symbol, Any}
)

    # extract params
    optim_options = Optim.Options(;params...)

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
        NelderMead(),
        optim_options
    )
    # update model parameters
    final_sol = Optim.minimizer(res)
    apply_params(model.forecast, final_sol)
    # return cost
    final_cost = minimum(res)
    return Solution(final_cost, final_sol)
end
