# basic model for testing gradient mode
model = ApplicationDrivenLearning.Model()
@variables(model, begin
    x >= 0, ApplicationDrivenLearning.Policy
    d, ApplicationDrivenLearning.Forecast
end)
@objective(ApplicationDrivenLearning.Plan(model), Min, x.plan)
@objective(ApplicationDrivenLearning.Assess(model), Min, x.assess)
set_optimizer(model, HiGHS.Optimizer)
set_silent(model)
ApplicationDrivenLearning.set_forecast_model(model, Chain(Dense(1 => 1)))
X = Float32.(ones(1, 1))
Y = Dict(d => Float32.(ones(1)))

@testset "GradientMode Stop Rules" begin
    # epochs
    initial_sol = ApplicationDrivenLearning.extract_params(model.forecast)
    opt = ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.GradientMode,
        epochs = 0,
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    @test initial_sol == sol.params

    # time_limit
    opt = ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.GradientMode,
        time_limit = 0,
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    @test initial_sol == sol.params

    # gradient norm
    initial_sol = ApplicationDrivenLearning.extract_params(model.forecast)
    opt = ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.GradientMode,
        g_tol = Inf,
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    @test initial_sol == sol.params
end

@testset "GradientMode Multi-Sample Gradient Correctness" begin
    # Newsvendor model where the per-sample cost-gradient (dC/dŷ) depends on
    # whether the forecast over- or under-shoots the realized demand. The
    # gradient that GradientMode applies to the parameters must equal the true
    # gradient of the aggregate cost, dC/dθ = (1/T) Σ_t (dC/dŷ_t)·(dŷ_t/dθ).
    #
    # The samples below are chosen so the over/under-stock regime correlates
    # with the input, i.e. mean_t(dCdy_t · J_t) ≠ mean_t(dCdy_t)·mean_t(J_t).
    # Aggregating dCdy across the batch *before* forming the surrogate loss
    # (mean(dCdy' * m(X'))) computes the right-hand side instead of the
    # left-hand side and is therefore biased for T > 1. This test pins the
    # left-hand side via finite differences of the true cost.
    c, q, r = 1.0, 3.0, 0.0
    nv = ApplicationDrivenLearning.Model()
    @variables(nv, begin
        x, ApplicationDrivenLearning.Policy
        d, ApplicationDrivenLearning.Forecast
    end)
    @variables(ApplicationDrivenLearning.Plan(nv), begin
        yp >= 0
        wp >= 0
    end)
    @constraints(ApplicationDrivenLearning.Plan(nv), begin
        yp <= d.plan
        yp + wp <= x.plan
    end)
    @objective(
        ApplicationDrivenLearning.Plan(nv),
        Min,
        c * x.plan - q * yp - r * wp
    )
    @variables(ApplicationDrivenLearning.Assess(nv), begin
        ya >= 0
        wa >= 0
    end)
    @constraints(ApplicationDrivenLearning.Assess(nv), begin
        ya <= d.assess
        ya + wa <= x.assess
    end)
    @objective(
        ApplicationDrivenLearning.Assess(nv),
        Min,
        c * x.assess - q * ya - r * wa
    )
    set_optimizer(nv, HiGHS.Optimizer)
    set_silent(nv)
    ApplicationDrivenLearning.set_forecast_model(
        nv,
        Chain(Dense(1 => 1; bias = false, init = (s...) -> ones(s...))),
    )

    # x = [1,2,3], θ₀ = 1 ⇒ ŷ = [1,2,3]; d = [5,5,1] ⇒ samples 1,2 understock
    # (dCdy = c-q = -2), sample 3 overstocks (dCdy = c = +1).
    # True gradient = mean(dCdy_t · x_t) = (-2·1 -2·2 +1·3)/3 = -1.
    # Aggregated/biased gradient = mean(dCdy_t)·mean(x_t) = (-1)·2 = -2.
    Xnv = reshape([1.0, 2.0, 3.0], 3, 1)
    Ynv = reshape([5.0, 5.0, 1.0], 3, 1)

    θ0 = ApplicationDrivenLearning.extract_params(nv.forecast)

    # Reference: central finite-difference gradient of the true aggregate cost.
    function cost_at(θ)
        ApplicationDrivenLearning.apply_params(nv.forecast, θ)
        return ApplicationDrivenLearning.compute_cost(nv, Xnv, Ynv)
    end
    h = 1e-4
    fd_grad = similar(θ0)
    for i in eachindex(θ0)
        θp = copy(θ0)
        θm = copy(θ0)
        θp[i] += h
        θm[i] -= h
        fd_grad[i] = (cost_at(θp) - cost_at(θm)) / (2h)
    end
    ApplicationDrivenLearning.apply_params(nv.forecast, θ0)

    # Gradient actually applied by the framework, recovered from a single
    # Descent(η) step: θ₁ = θ₀ - η·g  ⇒  g = (θ₀ - θ₁)/η.
    η = 1.0
    opt_state = Flux.setup(Flux.Descent(η), nv.forecast)
    _, dC = ApplicationDrivenLearning.compute_cost(nv, Xnv, Ynv, true)
    ApplicationDrivenLearning.apply_gradient!(nv.forecast, dC, Xnv, opt_state)
    θ1 = ApplicationDrivenLearning.extract_params(nv.forecast)
    applied_grad = (θ0 .- θ1) ./ η
    ApplicationDrivenLearning.apply_params(nv.forecast, θ0)

    @test applied_grad ≈ fd_grad atol = 1e-2
end
