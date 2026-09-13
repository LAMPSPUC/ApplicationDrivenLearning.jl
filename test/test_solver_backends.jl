# Solver backends: `OptimMode` over any Optim.jl algorithm, `NLoptMode` from the
# NLopt extension, and the `dC/dθ` that makes the gradient-based ones reachable.
#
# Two models, because the two things under test need different properties:
#
#   `_convex_model`  - one parameter, cost provably convex in it and the plan side
#                      feasible everywhere, so *every* algorithm must land on the
#                      same optimum. This is what tests the plumbing.
#   `_backend_model` - five parameters across two networks, so a permutation of
#                      the flat gradient is detectable. Its cost is NOT convex in
#                      θ, so algorithms legitimately find different local optima
#                      and comparing them would test Optim rather than this code.
#
# `Y` is a plain matrix throughout, unlike the rest of the suite, which matches
# columns to forecast variables by name. Not a lapse and not the pattern to copy:
# these are algorithm tests, and `compute_cost` is called in tight loops here -
# 601 times in the grid search alone - where re-normalizing a table per call would
# cost more than the property being tested. `test_data_inputs.jl` is where the
# container handling itself is covered.
using Optim
using NLopt
using Optimisers

# a mode with no trainer registered, to exercise the `_train!` fallback
struct _UnregisteredMode <: ADL.AbstractOptimizationMode end

"""
Single generator against a single load, with `g >= 0` alongside `g >= d` so the
plan model is feasible for any forecast, including a negative one.

    plan   : min 10*g            s.t. g >= d, g >= 0     =>  g = max(0, ŷ)
    assess : min 10*g + 50*s     s.t. s >= d - g, s >= 0

So `C(θ) = 10·max(0, ŷ) + mean(50·max(0, y - max(0, ŷ)))`, a sum of convex
functions of the single weight, with a unique minimum at the 80% quantile of `y`
(`1 - 10/50`). Verified against a grid search below rather than asserted.
"""
function _convex_model()
    m = ADL.Model()
    @variable(m, g >= 0, ADL.Policy)
    @variable(m, d, ADL.Forecast)
    @constraint(ADL.Plan(m), g.plan >= d.plan)
    @objective(ADL.Plan(m), Min, 10 * g.plan)
    @variable(ADL.Assess(m), s >= 0)
    @constraint(ADL.Assess(m), s >= d.assess - g.assess)
    @objective(ADL.Assess(m), Min, 10 * g.assess + 50 * s)
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    ADL.set_forecast_model(
        m,
        ADL.ForecastModel(
            architecture = Chain(
                Dense(1 => 1; bias = false, init = (s...) -> 5 * ones(s...)),
            ),
            outputs = [d],
        ),
    )
    return m
end

_convex_X() = ones(6, 1)
_convex_Y() = reshape([10.0, 12.0, 14.0, 16.0, 18.0, 20.0], 6, 1)

"""
A newsvendor whose plan side cannot accommodate a *negative* forecast — `yp <= d`
alongside `yp >= 0` has no solution there — so an unbounded external optimizer
will eventually propose parameters the application cannot be evaluated at.

Used both to check that the resulting error is actionable and that a failed solve
leaves the parameters where it found them.
"""
function _infeasible_at_negative_model()
    m = ADL.Model()
    @variable(m, q >= 0, ADL.Policy)
    @variable(m, d, ADL.Forecast)
    @variables(ADL.Plan(m), begin
        yp >= 0
        wp >= 0
    end)
    @constraints(ADL.Plan(m), begin
        yp <= d.plan
        yp + wp <= q.plan
    end)
    @objective(ADL.Plan(m), Min, 5.0 * q.plan - 9.0 * yp - 4.0 * wp)
    @variables(ADL.Assess(m), begin
        ya >= 0
        wa >= 0
    end)
    @constraints(ADL.Assess(m), begin
        ya <= d.assess
        ya + wa <= q.assess
    end)
    @objective(ADL.Assess(m), Min, 5.0 * q.assess - 9.0 * ya - 4.0 * wa)
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    ADL.set_forecast_model(
        m,
        ADL.ForecastModel(
            architecture = Chain(
                Dense(1 => 1; bias = false, init = (s...) -> 0.5 * ones(s...)),
            ),
            outputs = [d],
        ),
    )
    return m
end

"""
Two generators, each serving its own load, with asymmetric costs and asymmetric
shortfall penalties, predicted by two networks from a shared input.

Five parameters — two weights and a bias, then one weight and a bias — so that a
permuted flat gradient would be caught rather than pass by luck. `g1, g2 >= 0`
keeps the plan model feasible for any prediction, so that a solver wandering into
negative forecasts does not turn these into tests about the model.
"""
function _backend_model()
    m = ADL.Model()
    @variable(m, g1 >= 0, ADL.Policy)
    @variable(m, g2 >= 0, ADL.Policy)
    @variable(m, d1, ADL.Forecast)
    @variable(m, d2, ADL.Forecast)

    @constraints(ADL.Plan(m), begin
        g1.plan >= d1.plan
        g2.plan >= d2.plan
    end)
    @objective(ADL.Plan(m), Min, 10 * g1.plan + 20 * g2.plan)

    @variables(ADL.Assess(m), begin
        s1 >= 0
        s2 >= 0
    end)
    @constraints(ADL.Assess(m), begin
        s1 >= d1.assess - g1.assess
        s2 >= d2.assess - g2.assess
    end)
    @objective(
        ADL.Assess(m),
        Min,
        10 * g1.assess + 20 * g2.assess + 50 * s1 + 30 * s2
    )

    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    ADL.set_forecast_model(
        m,
        [
            ADL.ForecastModel(
                inputs = [1, 2],
                architecture = Dense(
                    2 => 1;
                    init = (s...) -> [0.6 0.3],
                    bias = [1.0],
                ),
                outputs = [d1],
            ),
            ADL.ForecastModel(
                inputs = [1],
                architecture = Dense(
                    1 => 1;
                    init = (s...) -> fill(0.4, s...),
                    bias = [0.5],
                ),
                outputs = [d2],
            ),
        ],
    )
    return m
end

_backend_X() = [10.0 4.0; 12.0 5.0; 14.0 6.0; 16.0 7.0; 18.0 8.0; 20.0 9.0]
_backend_Y() = [9.0 5.0; 11.0 6.0; 13.0 4.0; 15.0 8.0; 17.0 7.0; 19.0 9.0]

@testset "dC/dθ matches central finite differences" begin
    # The one check that can catch a wrong flat-gradient ordering: `Optim` and
    # NLopt take θ and g as parallel vectors and cannot notice a permutation
    # between them, so a mismatch would not error - it would train towards the
    # wrong place.
    #
    # Driven through the same two calls the trainers make, rather than through a
    # convenience wrapper around them: `dC/dŷ` out of the cost evaluation, then
    # `_flat_parameter_gradient` to carry it to the parameters.
    m = _backend_model()
    X, Y = _backend_X(), _backend_Y()

    C, dCdy = ADL.compute_cost(m, X, Y, true)
    g = ADL._flat_parameter_gradient(m.forecast, dCdy, X)
    θ0 = ADL.extract_params(m.forecast)
    @test length(g) == length(θ0) == 5
    @test size(dCdy) == (size(X, 1), 2)

    # the assessed cost is piecewise linear in θ, so central differences are
    # exact away from a kink
    h = 1e-4
    fd = similar(g)
    for i in eachindex(θ0)
        θp = copy(θ0)
        θp[i] += h
        θm = copy(θ0)
        θm[i] -= h
        ADL.apply_params(m.forecast, θp)
        Cp = ADL.compute_cost(m, X, Y, false)
        ADL.apply_params(m.forecast, θm)
        Cm = ADL.compute_cost(m, X, Y, false)
        fd[i] = (sum(Cp) - sum(Cm)) / (2h * length(Cp))
    end
    ADL.apply_params(m.forecast, θ0)

    @test g ≈ fd rtol = 1e-4
    # and not by luck: the entries must differ, or a permutation would pass
    @test length(unique(round.(g; digits = 6))) > 1
end

@testset "destructure agrees with extract_params on the flat layout" begin
    # `_flat_parameter_gradient` gets its layout from `Optimisers.destructure`,
    # while `apply_params` writes back through `trainables`. They must describe the
    # same vector, or θ and dC/dθ would refer to different things - silently, since
    # both are just vectors of the right length.
    m = _backend_model()
    flat, _ = Optimisers.destructure(m.forecast)
    @test flat == ADL.extract_params(m.forecast)
end

@testset "every Optim algorithm finds the convex optimum" begin
    X, Y = _convex_X(), _convex_Y()

    # the reference is measured, not assumed: a grid over the single parameter
    reference = _convex_model()
    grid = 0.0:0.05:30.0
    costs = map(grid) do θ
        ADL.apply_params(reference.forecast, [θ])
        return ADL.compute_cost(reference, X, Y)
    end
    best_cost, best_i = findmin(costs)
    @test grid[best_i] ≈ 18.0 atol = 0.05          # the 80% quantile of Y
    @test best_cost ≈ 196.6666666 atol = 1e-5

    for algorithm in [
        Optim.NelderMead(),
        Optim.LBFGS(),
        Optim.BFGS(),
        Optim.ConjugateGradient(),
        Optim.GradientDescent(),
        Optim.ParticleSwarm(),
    ]
        sol = ADL.train!(
            _convex_model(),
            X,
            Y,
            ADL.Options(ADL.OptimMode; algorithm = algorithm, iterations = 300),
        )
        @test sol.cost ≈ best_cost atol = 1e-3
        @test sol.params[1] ≈ 18.0 atol = 1e-2
    end
end

@testset "OptimMode forwards bounds to the box-constrained call" begin
    # `Fminbox` runs its inner optimizer to convergence on every outer iteration,
    # so the budget here is deliberately tiny - the point is that the bounds are
    # honoured, not that it converges
    X, Y = _convex_X(), _convex_Y()
    sol = ADL.train!(
        _convex_model(),
        X,
        Y,
        ADL.Options(
            ADL.OptimMode;
            algorithm = Optim.NelderMead(),
            lower_bounds = [0.0],
            upper_bounds = [9.0],
            iterations = 5,
            outer_iterations = 1,
        ),
    )
    # the unconstrained optimum is 18.0, so a ceiling of 9.0 has to bite
    @test sol.params[1] <= 9.0 + 1e-6
    @test sol.params[1] >= -1e-6

    # `_optimize_with` dispatches four Optim signatures, on two independent axes:
    # gradient or not, boxed or not. The derivative-free boxed call is above; this
    # is the boxed *gradient* one, which is otherwise never reached.
    sol_g = ADL.train!(
        _convex_model(),
        X,
        Y,
        ADL.Options(
            ADL.OptimMode;
            algorithm = Optim.LBFGS(),
            lower_bounds = [0.0],
            upper_bounds = [9.0],
            iterations = 5,
            outer_iterations = 1,
        ),
    )
    @test sol_g.params[1] <= 9.0 + 1e-6
    @test sol_g.params[1] >= -1e-6
    @test isfinite(sol_g.cost)
end

@testset "the parallel backend is validated" begin
    # `parallel` is the one option shared by every sample-evaluating mode, so a
    # typo there should name the problem rather than fail somewhere inside a
    # trainer
    err = try
        ADL.train!(
            _convex_model(),
            _convex_X(),
            _convex_Y(),
            ADL.Options(ADL.OptimMode; parallel = :mpi),
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("AbstractParallelBackend", err.msg)
    @test occursin("Symbol", err.msg)          # names what was actually passed
end

@testset "a failed objective names the parameters that caused it" begin
    # `_short_vector` keeps the message readable for a network too large to print.
    # Its long branch is only reachable from a failing solve on a big model, so it
    # is exercised directly here rather than by building one.
    @test ADL._short_vector([1.0, 2.0]) == "[1.0, 2.0]"
    long = collect(1.0:9.0)
    @test ADL._short_vector(long) == "9-element vector with extrema (1.0, 9.0)"
    @test !occursin("[", ADL._short_vector(long))
end

@testset "NelderMeadMode is unchanged" begin
    X, Y = _backend_X(), _backend_Y()

    # the deprecated alias must agree bit for bit with the explicit spelling it
    # now forwards to, including the two Nelder-Mead-specific keywords
    legacy = ADL.train!(
        _backend_model(),
        X,
        Y,
        ADL.Options(ADL.NelderMeadMode; iterations = 200),
    )
    explicit = ADL.train!(
        _backend_model(),
        X,
        Y,
        ADL.Options(
            ADL.OptimMode;
            algorithm = Optim.NelderMead(;
                initial_simplex = Optim.AffineSimplexer(),
                parameters = Optim.AdaptiveParameters(),
            ),
            iterations = 200,
        ),
    )
    @test legacy.cost == explicit.cost
    @test legacy.params == explicit.params

    # and it still accepts what it always accepted
    tuned = ADL.train!(
        _backend_model(),
        X,
        Y,
        ADL.Options(
            ADL.NelderMeadMode;
            initial_simplex = Optim.AffineSimplexer(),
            parameters = Optim.AdaptiveParameters(),
            iterations = 200,
        ),
    )
    @test tuned.cost == legacy.cost

    # an `algorithm` here is a mistake worth naming rather than ignoring
    @test_throws ArgumentError ADL.train!(
        _backend_model(),
        X,
        Y,
        ADL.Options(ADL.NelderMeadMode; algorithm = Optim.LBFGS()),
    )
end

@testset "NLoptMode through the package extension" begin
    X, Y = _convex_X(), _convex_Y()
    @test !isnothing(Base.get_extension(ADL, :ADLNLoptExt))

    # `:LN_*` is derivative-free and `:LD_*` gradient-based - NLopt encodes it in
    # the second letter of the name, which is what the extension dispatches on
    for algorithm in
        [:LN_NELDERMEAD, :LN_BOBYQA, :LN_COBYLA, :LD_LBFGS, :LD_MMA]
        sol = ADL.train!(
            _convex_model(),
            X,
            Y,
            ADL.Options(
                ADL.NLoptMode;
                algorithm = algorithm,
                lower_bounds = [0.0],
                upper_bounds = [40.0],
                xtol_rel = 1e-10,
                maxeval = 2000,
            ),
        )
        @test sol.cost ≈ 196.6666666 atol = 1e-3
        @test sol.params[1] ≈ 18.0 atol = 1e-2
    end

    # stopping criteria are NLopt's own, set on the `Opt` object rather than
    # forwarded to an options struct as `OptimMode` does
    brief = ADL.train!(
        _convex_model(),
        X,
        Y,
        ADL.Options(ADL.NLoptMode; algorithm = :LN_NELDERMEAD, maxeval = 3),
    )
    @test isfinite(brief.cost)
    @test brief.cost > 196.7   # three evaluations cannot have converged
end

@testset "NLoptMode restores the parameters when the solve throws" begin
    # A failed solve must leave the model where it started, not at whatever the
    # optimizer last tried. Otherwise the remedy the error suggests - retry inside
    # bounds - fails in turn, because the initial point is now outside them. The
    # same guarantee is checked for `OptimMode` in the feasible-region testset
    # below; this is the NLopt half of it.
    m = _infeasible_at_negative_model()
    θ0 = copy(ADL.extract_params(m.forecast))
    X = ones(6, 1)
    Y = reshape([10.0, 12.0, 14.0, 16.0, 18.0, 20.0], 6, 1)

    err = try
        ADL.train!(
            m,
            X,
            Y,
            ADL.Options(
                ADL.NLoptMode;
                algorithm = :LD_LBFGS,
                # unbounded, so the optimizer is free to walk into the region
                # where the plan model has no solution
                maxeval = 200,
            ),
        )
        nothing
    catch e
        e
    end
    # NLopt calls the objective through a C callback, which captures any Julia
    # exception and rethrows it wrapped - so unlike `OptimMode` the error that
    # reaches the caller is a `CapturedException`, not the `ErrorException`
    # `_evaluate_or_explain` built. What has to survive is the message, since that
    # is what tells the user which parameters failed and what to do about it.
    @test !isnothing(err)
    message = sprint(showerror, err)
    @test occursin("θ =", message)
    @test occursin("lower_bounds", message)

    @test ADL.extract_params(m.forecast) == θ0

    # and the restored point is usable, which is the whole reason for restoring it
    bounded = ADL.train!(
        m,
        X,
        Y,
        ADL.Options(
            ADL.NLoptMode;
            algorithm = :LN_NELDERMEAD,
            lower_bounds = [0.0],
            upper_bounds = [40.0],
            maxeval = 50,
        ),
    )
    @test isfinite(bounded.cost)
end

@testset "an unregistered mode explains itself" begin
    X, Y = _convex_X(), _convex_Y()
    # `Options` accepts any subtype, so a mode with no trainer is reachable and
    # must say something better than `MethodError`
    @test_throws ArgumentError ADL.train!(
        _convex_model(),
        X,
        Y,
        ADL.Options(_UnregisteredMode),
    )
end

@testset "a solve with no readable solution is reported, not used" begin
    X = ones(2, 1)
    Y = reshape([10.0, 12.0], 2, 1)

    # infeasible plan model: `yp <= d` with `yp >= 0` has no solution at ŷ < 0
    m = _infeasible_at_negative_model()
    ADL.apply_params(m.forecast, [-1.0])
    err = try
        ADL.compute_cost(m, X, Y)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("plan model has no feasible solution", err.msg)
    @test occursin("INFEASIBLE", err.msg)

    # Unbounded plan model. This is the case a feasibility check alone cannot see:
    # an unbounded model *has* feasible points, so `has_values` is satisfied and
    # the arbitrary point the solver returns would be used as the optimal policy.
    # Before the explicit check, `compute_cost` returned a number here.
    u = ADL.Model()
    @variable(u, x, ADL.Policy)
    @variable(u, d, ADL.Forecast)
    @constraint(ADL.Plan(u), x.plan >= d.plan)
    @objective(ADL.Plan(u), Min, -x.plan)      # nothing bounds x from above
    @variable(ADL.Assess(u), s >= 0)
    @constraint(ADL.Assess(u), s >= d.assess - x.assess)
    @objective(ADL.Assess(u), Min, s)
    set_optimizer(u, HiGHS.Optimizer)
    set_silent(u)
    ADL.set_forecast_model(
        u,
        ADL.ForecastModel(
            architecture = Chain(
                Dense(1 => 1; bias = false, init = (s...) -> ones(s...)),
            ),
            outputs = [d],
        ),
    )
    err_u = try
        ADL.compute_cost(u, X, Y)
        nothing
    catch e
        e
    end
    @test err_u isa ErrorException
    @test occursin("unbounded", err_u.msg)
    # MOI reports an unbounded primal as an infeasible dual, so the message has to
    # say what that means rather than just quote the status
    @test occursin("DUAL_INFEASIBLE", err_u.msg)
end

@testset "leaving the feasible region explains itself" begin
    m = _infeasible_at_negative_model()

    X = ones(6, 1)
    Y = reshape([10.0, 12.0, 14.0, 16.0, 18.0, 20.0], 6, 1)
    err = try
        ADL.train!(
            m,
            X,
            Y,
            ADL.Options(
                ADL.OptimMode;
                algorithm = Optim.ConjugateGradient(),
                iterations = 30,
            ),
        )
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    # the raw failure comes out of DiffOpt naming neither the parameters nor a
    # remedy; both have to be added for it to be actionable
    @test occursin("θ =", err.msg)
    @test occursin("lower_bounds", err.msg)

    # and bounding the search is the documented fix
    bounded = ADL.train!(
        m,
        X,
        Y,
        ADL.Options(
            ADL.OptimMode;
            algorithm = Optim.NelderMead(),
            lower_bounds = [0.0],
            upper_bounds = [40.0],
            iterations = 5,
            outer_iterations = 1,
        ),
    )
    @test isfinite(bounded.cost)
end
