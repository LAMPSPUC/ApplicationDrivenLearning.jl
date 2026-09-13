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

"""
Newsvendor with `c < q` and `r = 0`, so the per-sample cost gradient with
respect to the forecast is `c - q` when the forecast understocks and `c` when
it overstocks. Returns the model and its `Forecast` variable.
"""
function _sgd_newsvendor(c_s = 1.0, q_s = 3.0, r_s = 0.0)
    m = ApplicationDrivenLearning.Model()
    @variables(m, begin
        x, ApplicationDrivenLearning.Policy
        d, ApplicationDrivenLearning.Forecast
    end)
    @variables(ApplicationDrivenLearning.Plan(m), begin
        yp >= 0
        wp >= 0
    end)
    @constraints(ApplicationDrivenLearning.Plan(m), begin
        yp <= d.plan
        yp + wp <= x.plan
    end)
    @objective(
        ApplicationDrivenLearning.Plan(m),
        Min,
        c_s * x.plan - q_s * yp - r_s * wp
    )
    @variables(ApplicationDrivenLearning.Assess(m), begin
        ya >= 0
        wa >= 0
    end)
    @constraints(ApplicationDrivenLearning.Assess(m), begin
        ya <= d.assess
        ya + wa <= x.assess
    end)
    @objective(
        ApplicationDrivenLearning.Assess(m),
        Min,
        c_s * x.assess - q_s * ya - r_s * wa
    )
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    ApplicationDrivenLearning.set_forecast_model(
        m,
        Chain(Dense(1 => 1; bias = false, init = (s...) -> ones(s...))),
    )
    return m, d
end

@testset "GradientMode Stochastic Batches" begin
    # The stochastic branch of `_train_with_gradient!` and the whole of
    # `_stochastic_compute` are otherwise only reached from the `mpiexec`
    # subprocess in test_mpi.jl, which is skipped whenever ADL_SKIP_MPI_TESTS
    # is set and never reports coverage.
    c_s, q_s = 1.0, 3.0

    # `_stochastic_compute` must take its gradient from the batch and, when
    # asked for the full cost, its cost from the whole dataset.
    ms, _ = _sgd_newsvendor(c_s, q_s)

    # θ₀ = 1 and no bias, so ŷ_t = x_t. Samples 1 and 2 understock
    # (ŷ < d ⇒ dC/dŷ = c - q = -2), samples 3 and 4 overstock
    # (ŷ > d ⇒ dC/dŷ = c = 1).
    Xs = reshape([1.0, 2.0, 3.0, 4.0], 4, 1)
    Ys = reshape([9.0, 9.0, 1.0, 1.0], 4, 1)
    batch = [1, 3]
    epochx = Xs[batch, :]

    C_batch, dC_batch = ApplicationDrivenLearning._stochastic_compute(
        ms,
        Xs,
        Ys,
        epochx,
        batch,
        false,
    )
    # per-sample assessed costs are [-2, -4, 0, 1]
    @test C_batch ≈ -1.0                     # mean over the batch
    @test size(dC_batch) == (length(batch), 1)
    # rows follow the batch, not the dataset: sample 1 understocks, 3 overstocks
    @test vec(dC_batch) ≈ [c_s - q_s, c_s]

    C_full, dC_full = ApplicationDrivenLearning._stochastic_compute(
        ms,
        Xs,
        Ys,
        epochx,
        batch,
        true,
    )
    @test C_full ≈ -1.25                     # mean over the whole dataset
    @test C_full != C_batch                  # the two really do differ here
    @test dC_full == dC_batch                # gradient still only over the batch

    # end to end: the stochastic branch trains to the known optimum. All four
    # samples share the same demand, so the run is independent of which rows
    # each batch happens to draw.
    me, de = _sgd_newsvendor(c_s, q_s)
    Xe = ones(4, 1)
    Ye = Dict(de => fill(50.0, 4))
    Random.seed!(2024)
    sol = ApplicationDrivenLearning.train!(
        me,
        Xe,
        Ye,
        ApplicationDrivenLearning.Options(
            ApplicationDrivenLearning.GradientMode;
            rule = Flux.Adam(1.0),
            epochs = 200,
            batch_size = 2,
            verbose = false,
        ),
    )
    @test sol.params[1] ≈ 50.0 atol = 1e-1
    @test sol.cost ≈ (c_s - q_s) * 50.0 atol = 1e-1

    # `compute_cost_every` gates only the extra full-dataset cost sweep, so
    # training still runs when it never fires
    mq, dq = _sgd_newsvendor(c_s, q_s)
    Random.seed!(2024)
    sol_q = ApplicationDrivenLearning.train!(
        mq,
        Xe,
        Dict(dq => fill(50.0, 4)),
        ApplicationDrivenLearning.Options(
            ApplicationDrivenLearning.GradientMode;
            rule = Flux.Adam(1.0),
            epochs = 3,
            batch_size = 2,
            compute_cost_every = 10,
            verbose = false,
        ),
    )
    @test sol_q.cost == Inf                  # no full cost was ever computed
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
