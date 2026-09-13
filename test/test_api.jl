ADL = ApplicationDrivenLearning

@testset "Options validation" begin
    opt = ADL.Options(ADL.GradientMode; epochs = 5)
    @test opt.mode === ADL.GradientMode
    @test opt.params[:epochs] == 5

    # a non-mode type and a mode *instance* must both be rejected
    @test_throws ArgumentError ADL.Options(Int)
    @test_throws ArgumentError ADL.Options(ADL.GradientMode())
end

@testset "Forecast variables reject bounds" begin
    m = ADL.Model()
    # `@test_logs` evaluates its expression in a closure, so the variable has
    # to be taken from the returned value rather than from the macro binding
    fl =
        @test_logs (:warn, "Forecast variable lower bound will be removed.") (@variable(
            m,
            fl_var >= 0,
            ADL.Forecast
        ))
    @test !JuMP.has_lower_bound(fl.plan)
    @test !JuMP.has_lower_bound(fl.assess)

    fu =
        @test_logs (:warn, "Forecast variable upper bound will be removed.") (@variable(
            m,
            fu_var <= 10,
            ADL.Forecast
        ))
    @test !JuMP.has_upper_bound(fu.plan)
    @test !JuMP.has_upper_bound(fu.assess)
end

@testset "Array property access preserves shape" begin
    m = ADL.Model()
    @variable(m, w[1:2, 1:3], ADL.Policy)
    @test size(w.plan) == (2, 3)
    @test size(w.assess) == (2, 3)
    @test w.plan[2, 3] == w[2, 3].plan
    @test w.assess[1, 2] == w[1, 2].assess

    @variable(m, g[1:2, 1:2], ADL.Forecast)
    @test size(g.plan) == (2, 2)
    @test g.plan[2, 1] == g[2, 1].plan

    # non 1-based axes must keep their axes, so that `v.plan[i] == v[i].plan`
    @variable(m, v[i = 2:4], ADL.Policy)
    @test v.plan[2] == v[2].plan
    @test v.plan[4] == v[4].plan
    @test v.assess[3] == v[3].assess
end

@testset "flux_utils" begin
    dense = Flux.Dense(2 => 1) |> f64
    θ = ADL._extract_flux_params(dense)
    @test length(θ) == 3

    ADL._fix_flux_params_single_model(dense, [1.0, 2.0, 3.0])
    @test dense.weight == [1.0 2.0]
    @test dense.bias == [3.0]

    @test ADL._has_params(dense)
    @test ADL._has_params(Flux.Chain(Flux.Dense(2 => 1)))
    @test !ADL._has_params(Flux.relu)
    @test !ADL._has_params(identity)

    # the helpers must work on any Flux layer, not just Dense and Chain
    scale = Flux.Scale(2) |> f64
    @test ADL._has_params(scale)
    @test length(ADL._extract_flux_params(scale)) == 4
    ADL._fix_flux_params_single_model(scale, [2.0, 3.0, 4.0, 5.0])
    @test ADL._extract_flux_params(scale) == [2.0, 3.0, 4.0, 5.0]
    @test scale([1.0, 1.0]) ≈ [2.0 + 4.0, 3.0 + 5.0]
end

@testset "PredictiveModel from Chain with input_output_map" begin
    m = ADL.Model()
    @variable(m, f[1:2], ADL.Forecast)
    chain = Flux.Chain(Flux.Dense(2 => 1)) |> f64
    iomap = Dict([1, 2] => [f[1]], [1, 3] => [f[2]])
    pm = ADL.PredictiveModel(chain, iomap)
    @test pm.input_size == 3
    @test pm.output_size == 2
    @test size(pm(ones(3, 4))) == (2, 4)
    @test length(pm(ones(3))) == 2

    # mismatching map sizes must be rejected
    @test_throws AssertionError ADL.PredictiveModel(chain, Dict([1] => [f[1]]))
end

@testset "PredictiveModel with heterogeneous networks" begin
    m = ADL.Model()
    @variable(m, f[1:3], ADL.Forecast)
    # a Dense and a Chain in the same predictive model
    nets = Any[Flux.Dense(2 => 1)|>f64, Flux.Chain(Flux.Dense(1 => 2))|>f64]
    iomap = [Dict([1, 2] => [f[1]]), Dict([3] => [f[2], f[3]])]
    pm = ADL.PredictiveModel(nets, iomap)
    @test pm.input_size == 3
    @test pm.output_size == 3
    @test size(pm(ones(3, 5))) == (3, 5)
    @test length(pm(ones(3))) == 3

    # parameters of both networks must round-trip
    θ = ADL.extract_params(pm)
    @test length(θ) == (2 * 1 + 1) + (1 * 2 + 2)
    ADL.apply_params(pm, ones(length(θ)))
    @test ADL.extract_params(pm) == ones(length(θ))
end

@testset "PredictiveModel with non-Dense layer types" begin
    m = ADL.Model()
    @variable(m, f[1:4], ADL.Forecast)

    # none of these is a plain Dense: a Chain with a bare activation function
    # as a layer, a Chain with a fused activation, and a Flux.Scale
    chain_bare =
        Flux.Chain(Flux.Dense(1 => 3), Flux.relu, Flux.Dense(3 => 1)) |> f64
    chain_fused =
        Flux.Chain(Flux.Dense(2 => 3, tanh), Flux.Dense(3 => 1)) |> f64
    scale = Flux.Scale(2) |> f64

    nets = Any[chain_bare, chain_fused, scale]
    iomap = [
        Dict([1] => [f[1]]),
        Dict([2, 3] => [f[2]]),
        Dict([1, 4] => [f[3], f[4]]),
    ]
    pm = ADL.PredictiveModel(nets, iomap)
    @test pm.input_size == 4
    @test pm.output_size == 4

    Xm = reshape(collect(1.0:12.0), 4, 3)
    Ym = pm(Xm)
    @test size(Ym) == (4, 3)

    # the matrix and the vector call paths are separate implementations and
    # must agree column by column
    for j = 1:3
        @test pm(Xm[:, j]) ≈ Ym[:, j]
    end

    # parameter layout: chain_bare | chain_fused | scale
    n1 = (1 * 3 + 3) + (3 * 1 + 1)   # 10
    n2 = (2 * 3 + 3) + (3 * 1 + 1)   # 13
    n3 = 2 + 2                       # scale weight + bias
    nparams = n1 + n2 + n3
    @test length(ADL.extract_params(pm)) == nparams

    ADL.apply_params(pm, fill(0.1, nparams))
    before = copy(ADL.extract_params(pm))
    @test before == fill(0.1, nparams)

    # a gradient step must reach every network, the Scale included
    ADL.apply_gradient!(
        pm,
        ones(3, 4),
        permutedims(Xm),
        Flux.setup(Flux.Descent(0.1), pm),
    )
    after = ADL.extract_params(pm)
    @test length(after) == nparams
    @test after[1:n1] != before[1:n1]
    @test after[(n1+1):(n1+n2)] != before[(n1+1):(n1+n2)]
    @test after[(n1+n2+1):end] != before[(n1+n2+1):end]
end

@testset "set_forecast_model size check" begin
    m = ADL.Model()
    @variable(m, f[1:2], ADL.Forecast)
    @test_throws AssertionError ADL.set_forecast_model(m, Flux.Dense(1 => 3))
    ADL.set_forecast_model(m, Flux.Dense(1 => 2))
    @test m.forecast.output_variables == m.forecast_vars
end

# --- a small newsvendor used by the remaining test sets -------------------
function _build_newsvendor()
    m = ADL.Model()
    @variables(m, begin
        x, ADL.Policy
        d, ADL.Forecast
    end)
    @variables(ADL.Plan(m), begin
        yp >= 0
        wp >= 0
    end)
    @constraints(ADL.Plan(m), begin
        yp <= d.plan
        yp + wp <= x.plan
    end)
    @objective(ADL.Plan(m), Min, 5.0 * x.plan - 9.0 * yp - 4.0 * wp)
    @variables(ADL.Assess(m), begin
        ya >= 0
        wa >= 0
    end)
    @constraints(ADL.Assess(m), begin
        ya <= d.assess
        ya + wa <= x.assess
    end)
    @objective(ADL.Assess(m), Min, 5.0 * x.assess - 9.0 * ya - 4.0 * wa)
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    return m, d
end

@testset "compute_cost aggregate flag" begin
    m, d = _build_newsvendor()
    ADL.set_forecast_model(
        m,
        Chain(Dense(1 => 1; bias = false, init = (s...) -> ones(s...))),
    )
    Xc = reshape([10.0, 20.0], 2, 1)
    Yc = Dict(d => [10.0, 20.0])

    agg = ADL.compute_cost(m, Xc, Yc)
    per_sample = ADL.compute_cost(m, Xc, Yc, false, false)
    @test per_sample isa Vector
    @test length(per_sample) == 2
    @test agg ≈ sum(per_sample) / 2

    # perfect forecast on a newsvendor pays (c - q) * d
    @test per_sample ≈ [(5.0 - 9.0) * 10.0, (5.0 - 9.0) * 20.0] atol = 1e-6

    # errors before a forecast model is set
    m2, _ = _build_newsvendor()
    @test_throws ArgumentError ADL.compute_cost(m2, Xc, reshape(Yc[d], 2, 1))
end

@testset "train! accepts a matrix or a dictionary for Y" begin
    Xt = ones(1, 1)
    Yvec = [50.0]
    opt() = ADL.Options(ADL.NelderMeadMode, iterations = 60, show_trace = false)

    m_dict, d_dict = _build_newsvendor()
    ADL.set_forecast_model(
        m_dict,
        Chain(Dense(1 => 1; bias = false, init = (s...) -> ones(s...))),
    )
    sol_dict = ADL.train!(m_dict, Xt, Dict(d_dict => Yvec), opt())

    m_mat, _ = _build_newsvendor()
    ADL.set_forecast_model(
        m_mat,
        Chain(Dense(1 => 1; bias = false, init = (s...) -> ones(s...))),
    )
    sol_mat = ADL.train!(m_mat, Xt, reshape(Yvec, 1, 1), opt())

    @test sol_mat.cost ≈ sol_dict.cost atol = 1e-6
    @test sol_mat.params ≈ sol_dict.params atol = 1e-6

    m_unset, _ = _build_newsvendor()
    @test_throws ArgumentError ADL.train!(
        m_unset,
        Xt,
        reshape(Yvec, 1, 1),
        opt(),
    )
end

@testset "Solution stores concrete element types" begin
    m, d = _build_newsvendor()
    ADL.set_forecast_model(
        m,
        Chain(Dense(1 => 1; bias = false, init = (s...) -> ones(s...))),
    )
    sol = ADL.train!(
        m,
        ones(1, 1),
        Dict(d => [50.0]),
        ADL.Options(ADL.NelderMeadMode, iterations = 30, show_trace = false),
    )
    @test sol isa ADL.Solution
    @test isconcretetype(typeof(sol.cost))
    @test isconcretetype(eltype(sol.params))
    @test eltype(sol.params) <: AbstractFloat
end

@testset "BilevelMode after compute_cost has built the model" begin
    # `_build` adds the `assess_policy_fix` constraints to the assess model.
    # `_solve_bilevel` must skip them when copying the assess constraints into
    # the upper level, otherwise the policy is pinned to zero.
    m, d = _build_newsvendor()
    ADL.set_forecast_model(
        m,
        Chain(Dense(1 => 1; bias = false, init = (s...) -> ones(s...))),
    )
    Xb = ones(1, 1)
    Yb = Dict(d => [50.0])

    # force the model to be built before training
    ADL.compute_cost(m, Xb, Yb)
    @test m.build

    opt = ADL.Options(
        ADL.BilevelMode,
        optimizer = HiGHS.Optimizer,
        mode = BilevelJuMP.FortunyAmatMcCarlMode(
            primal_big_M = 1000,
            dual_big_M = 1000,
        ),
        silent = true,
    )
    sol = ADL.train!(m, Xb, Yb, opt)
    @test sol.params[1] ≈ 50.0 atol = 1e-2
    @test sol.cost ≈ (5.0 - 9.0) * 50.0 atol = 1e-2
end

@testset "Options are not consumed by train!" begin
    # running the same Options object twice must give the same result
    m, d = _build_newsvendor()
    ADL.set_forecast_model(
        m,
        Chain(Dense(1 => 1; bias = false, init = (s...) -> 0.5 .* ones(s...))),
    )
    Xn = ones(1, 1)
    Yn = Dict(d => [50.0])
    opt = ADL.Options(ADL.NelderMeadMode, iterations = 50, show_trace = false)
    keys_before = sort(collect(keys(opt.params)))
    sol1 = ADL.train!(m, Xn, Yn, opt)
    @test sort(collect(keys(opt.params))) == keys_before

    ADL.set_forecast_model(
        m,
        Chain(Dense(1 => 1; bias = false, init = (s...) -> 0.5 .* ones(s...))),
    )
    sol2 = ADL.train!(m, Xn, Yn, opt)
    @test sol1.cost ≈ sol2.cost atol = 1e-6
end

@testset "JuMP interface on ApplicationDrivenLearning.Model" begin
    m, d = _build_newsvendor()
    ADL.set_forecast_model(m, Chain(Dense(1 => 1)))

    @test JuMP.objective_sense(m) == MOI.MIN_SENSE
    @test JuMP.num_variables(m) ==
          JuMP.num_variables(m.plan) + JuMP.num_variables(m.assess)
    for flag in (true, false)
        @test JuMP.num_constraints(
            m;
            count_variable_in_set_constraints = flag,
        ) ==
              JuMP.num_constraints(
            m.plan;
            count_variable_in_set_constraints = flag,
        ) + JuMP.num_constraints(
            m.assess;
            count_variable_in_set_constraints = flag,
        )
    end
    @test JuMP.object_dictionary(m) === m.obj_dict

    # printing must go to the given IO and must not error
    out = sprint(print, m)
    @test occursin("Plan Model:", out)
    @test occursin("Assess Model:", out)
    @test occursin("Forecast Model:", out)

    m_no_forecast, _ = _build_newsvendor()
    @test occursin("Not defined.", sprint(print, m_no_forecast))

    io = IOBuffer()
    JuMP.show_constraints_summary(io, m)
    @test occursin("Plan Model:", String(take!(io)))

    io = IOBuffer()
    JuMP.show_backend_summary(io, m)
    @test occursin("Plan Model:", String(take!(io)))
end

@testset "_dict_to_var_indexed_matrix" begin
    m = ADL.Model()
    @variable(m, f[1:2], ADL.Forecast)
    data = Dict(f[1] => [1.0, 2.0], f[2] => [3.0, 4.0])
    @test ADL._dict_to_var_indexed_matrix(data, [f[1], f[2]]) ==
          [1.0 3.0; 2.0 4.0]
    # column order follows row_index, not insertion order
    @test ADL._dict_to_var_indexed_matrix(data, [f[2], f[1]]) ==
          [3.0 1.0; 4.0 2.0]
    @test_throws ArgumentError ADL._dict_to_var_indexed_matrix(
        Dict(f[1] => [1.0, 2.0], f[2] => [3.0]),
        [f[1], f[2]],
    )
    # a variable with no series at all is named, rather than raising a `KeyError`
    @test_throws ArgumentError ADL._dict_to_var_indexed_matrix(
        Dict(f[1] => [1.0, 2.0]),
        [f[1], f[2]],
    )
end
