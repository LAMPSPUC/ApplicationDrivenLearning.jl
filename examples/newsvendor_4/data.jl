Random.seed!(0)

function generate_inar_p_poisson(T::Int, p::Int, init_delta::Int=200)
    α = rand(Distributions.Uniform(0.1, 1.0), p)
    α = 0.9 * α ./ sum(α)  # Normalize to sum to 1
    λ = 10.0
    T = T + init_delta
    demand = Vector{Float32}(undef, T)

    # Initialize first p values with Poisson noise
    for t in 1:p
        demand[t] = rand(Distributions.Poisson(λ))
    end

    # Generate demand series using INAR(p) recursion
    for t in (p+1):T
        retained = sum(rand(Distributions.Binomial(demand[t-i], α[i])) for i in 1:p)  # Binomial thinning
        innovation = rand(Distributions.Poisson(λ))  # Poisson-distributed noise
        demand[t] = retained + innovation
    end

    return demand[init_delta+1:end]
end

function generate_ar_p(T::Int, p::Int, μ::Float64=10.0, σ::Float64=3.0)
    α = rand(Distributions.Uniform(0.1, 1.0), p)
    α = 1.0 * α ./ sum(α)  # Normalize to sum to 1
    demand = Vector{Float32}(undef, T)

    # Initialize first p values with Gaussian noise
    for t in 1:p
        demand[t] = rand(Distributions.Normal(μ, σ))
    end

    # Generate demand series using AR(p) recursion
    for t in (p+1):T
        demand[t] = (
            sum(α[i] * demand[t-i]^2 for i in 1:p)^0.5 + 
            rand(Distributions.Normal(0.0, σ))
        )
    end

    return demand
end

function generate_series_data(I::Int, T::Int, r::Int, p::Int)
    Y = Matrix{Float32}(undef, T+p, I)
    for i=1:I
        Y[:, i] = max.(0.1, generate_ar_p(T+p, r))
    end
    X = Matrix{Float32}(undef, T, I*p)
    for t=1:T
        for ip=1:p
            for i=1:I
                X[t, p*(i-1) + ip] = Y[t+ip-1, i]
            end
        end
    end
    Y = Y[p+1:end, :]
    return X, Y
end

"""
Para I=2:
    1)
        acima: -μ+(μ-1) = -1
        abaixo: μ-(2μ-1) = -(μ-1)
    2)
        acima: -μ+1 = -(μ-1)
        abaixo: μ - (μ+1) = -1
"""
function generate_problem_data(I, μ=10.0)
    if I == 2
        c = [μ, μ]
        q = [2*μ-1, μ + 1]
        r = [μ - 1, 1.0]
    else
        c = μ * ones(I)
        r = rand((1.0, μ-1), I)
        q = c .+ r
    end

    return c, q, r
end