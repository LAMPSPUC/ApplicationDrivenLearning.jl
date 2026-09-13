c = 5.0
q = 9.0
r = 4.0

model = ApplicationDrivenLearning.Model()
@variables(model, begin
    x, ApplicationDrivenLearning.Policy
    d, ApplicationDrivenLearning.Forecast
end)
@variables(ApplicationDrivenLearning.Plan(model), begin
    y >= 0
    w >= 0
end)
@constraints(ApplicationDrivenLearning.Plan(model), begin
    con1, y <= d.plan
    con2, y + w <= x.plan
end)
@objective(
    ApplicationDrivenLearning.Plan(model),
    Min,
    c * x.plan - q * y - r * w
)
@variables(ApplicationDrivenLearning.Assess(model), begin
    y >= 0
    w >= 0
end)
@constraints(
    ApplicationDrivenLearning.Assess(model),
    begin
        con1, y <= d.assess
        con2, y + w <= x.assess
    end
)
@objective(
    ApplicationDrivenLearning.Assess(model),
    Min,
    c * x.assess - q * y - r * w
)
set_optimizer(model, HiGHS.Optimizer)
set_silent(model)
nn = Chain(Dense(1 => 1; bias = false, init = (size...) -> rand(size...)))

X = ones(1, 1)
Y = Dict(d => [50.0])
best_decision = y = Y[d][1]
best_cost = (c - q) * y

@testset "Newsvendor BilevelMode" begin
    ApplicationDrivenLearning.set_forecast_model(
        model,
        ApplicationDrivenLearning.ForecastModel(
            architecture = Chain(
                Dense(1 => 1; bias = false, init = (size...) -> rand(size...)),
            ),
            outputs = [d],
        ),
    )
    opt = ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.BilevelMode,
        optimizer = HiGHS.Optimizer,
        mode = BilevelJuMP.FortunyAmatMcCarlMode(
            primal_big_M = 100,
            dual_big_M = 100,
        ),
        silent = true,
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    @test sol.params[1] ≈ best_decision atol = 1e-2
    @test sol.cost ≈ best_cost atol = 1e-2
    @test ApplicationDrivenLearning.compute_cost(model, X, Y) ≈ best_cost atol =
        1e-2
end

"""
Two-sample newsvendor whose optimal order quantity differs per sample, so a
`Dense(1 => 1)` *with* a bias is exactly identified: `ŷ(1) = 50` and `ŷ(2) = 60`
force `w = 10` and `b = 40`.

Built fresh rather than reusing `model` above, whose single sample would leave
`w` and `b` underdetermined — and an upper-level objective that depends only on
`w + b` gives the bilevel MIP an unbounded ray.
"""
function _biased_newsvendor(; bounded::Bool = false, trailing_identity = false)
    m = ApplicationDrivenLearning.Model()
    @variables(m, begin
        x, ApplicationDrivenLearning.Policy
        d, ApplicationDrivenLearning.Forecast
    end)
    # `bounded` puts an *upper* bound on the inner variables. Every other bilevel
    # test declares `y >= 0` only, which leaves the bound-propagation branches of
    # the reformulation unexercised.
    if bounded
        @variables(ApplicationDrivenLearning.Plan(m), begin
            0 <= yp <= 100
            0 <= wp <= 100
        end)
        @variables(ApplicationDrivenLearning.Assess(m), begin
            0 <= ya <= 100
            0 <= wa <= 100
        end)
    else
        @variables(ApplicationDrivenLearning.Plan(m), begin
            yp >= 0
            wp >= 0
        end)
        @variables(ApplicationDrivenLearning.Assess(m), begin
            ya >= 0
            wa >= 0
        end)
    end
    @constraints(ApplicationDrivenLearning.Plan(m), begin
        yp <= d.plan
        yp + wp <= x.plan
    end)
    @objective(
        ApplicationDrivenLearning.Plan(m),
        Min,
        c * x.plan - q * yp - r * wp
    )
    @constraints(ApplicationDrivenLearning.Assess(m), begin
        ya <= d.assess
        ya + wa <= x.assess
    end)
    @objective(
        ApplicationDrivenLearning.Assess(m),
        Min,
        c * x.assess - q * ya - r * wa
    )
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    # `|> f64` as elsewhere in the suite: `Dense` defaults to Float32 parameters,
    # and feeding them Float64 samples warns on every forward pass
    layers = trailing_identity ? (Dense(1 => 1), identity) : (Dense(1 => 1),)
    ApplicationDrivenLearning.set_forecast_model(
        m,
        ApplicationDrivenLearning.ForecastModel(
            architecture = Chain(layers...) |> f64,
            outputs = [d],
        ),
    )
    return m, d
end

function _biased_opt()
    return ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.BilevelMode,
        optimizer = HiGHS.Optimizer,
        mode = BilevelJuMP.FortunyAmatMcCarlMode(
            primal_big_M = 1000,
            dual_big_M = 100,
        ),
        silent = true,
    )
end

@testset "Newsvendor BilevelMode with a bias" begin
    # Every other BilevelMode test uses `bias = false`, which skips the branch
    # that makes `b` an upper-level variable and the one that writes its solved
    # value back - so the *default* `Dense(n => m)` was untested here.
    mb, db = _biased_newsvendor()
    Xb = reshape([1.0, 2.0], 2, 1)
    Yb = Dict(db => [50.0, 60.0])

    sol = ApplicationDrivenLearning.train!(mb, Xb, Yb, _biased_opt())

    # c < q and r < c, so the optimum orders exactly the realized demand and the
    # per-sample cost is (c - q) * d
    @test sol.cost ≈ (c - q) * 55.0 atol = 1e-2

    # the bias really was solved for and written back, not left at its
    # initialization: prediction must interpolate both samples
    layer = mb.forecast.units[1].architecture[1]
    @test only(layer.weight) ≈ 10.0 atol = 1e-2
    @test only(layer.bias) ≈ 40.0 atol = 1e-2
    @test vec(mb.forecast(Xb')) ≈ [50.0, 60.0] atol = 1e-2
end

@testset "Newsvendor BilevelMode with bounded variables and a bare layer" begin
    # Two more reformulation branches: propagating an upper bound onto the
    # lower/upper-level copies of a variable, and a Chain element that is a plain
    # function rather than a layer. Only identity-like functions survive being
    # applied to a JuMP expression, which is the practical limit of that branch.
    mb, db = _biased_newsvendor(; bounded = true, trailing_identity = true)
    Xb = reshape([1.0, 2.0], 2, 1)
    Yb = Dict(db => [50.0, 60.0])

    sol = ApplicationDrivenLearning.train!(mb, Xb, Yb, _biased_opt())

    @test sol.cost ≈ (c - q) * 55.0 atol = 1e-2
    @test vec(mb.forecast(Xb')) ≈ [50.0, 60.0] atol = 1e-2
end

@testset "BilevelMode refuses a layer it cannot reformulate" begin
    # `BilevelMode` rebuilds the network symbolically, so a layer it does not know
    # how to express has no reformulation. It used to print a line and carry on,
    # which solved a *different* network than the one being trained and reported
    # the answer as if it were the right one.
    mb, db = _biased_newsvendor()
    ApplicationDrivenLearning.set_forecast_model(
        mb,
        ApplicationDrivenLearning.ForecastModel(
            architecture = Chain(Dense(1 => 1), Dropout(0.5)) |> f64,
            outputs = [db],
        ),
    )
    err = try
        ApplicationDrivenLearning.train!(
            mb,
            reshape([1.0, 2.0], 2, 1),
            Dict(db => [50.0, 60.0]),
            _biased_opt(),
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Dropout", err.msg)
    # and it points at the modes that do not need a symbolic rebuild
    @test occursin("GradientMode", err.msg)
end

@testset "Newsvendor OptimMode" begin
    ApplicationDrivenLearning.set_forecast_model(
        model,
        ApplicationDrivenLearning.ForecastModel(
            architecture = Chain(
                Dense(1 => 1; bias = false, init = (size...) -> rand(size...)),
            ),
            outputs = [d],
        ),
    )
    opt = ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.OptimMode,
        algorithm = Optim.NelderMead(),
        iterations = 100,
        time_limit = 60,
        show_trace = false,
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    @test sol.params[1] ≈ best_decision atol = 1e-2
    @test sol.cost ≈ best_cost atol = 1e-2
end

@testset "Newsvendor GradientMode" begin
    ApplicationDrivenLearning.set_forecast_model(
        model,
        ApplicationDrivenLearning.ForecastModel(
            architecture = Chain(
                Dense(1 => 1; bias = false, init = (size...) -> rand(size...)),
            ),
            outputs = [d],
        ),
    )
    opt = ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.GradientMode;
        rule = Flux.Adam(1.0),
        epochs = 200,
        batch_size = -1,
        verbose = false,
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    @test sol.params[1] ≈ best_decision atol = 1e-2
    @test sol.cost ≈ best_cost atol = 1e-2
end
