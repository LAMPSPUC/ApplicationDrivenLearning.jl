# Input containers accepted by `compute_cost` and `train!`: matrices, vectors,
# Tables.jl tables (`DataFrame`, `NamedTuple` of vectors, ...) and the
# `Dict{Forecast,Vector}` form.
#
# The property under test throughout is the rule stated at the top of
# `src/data.jl`: a named container is matched by name, an unnamed one by
# position, and neither is ever guessed at. So the tests come in pairs - one that
# a correct-but-reordered table still gives the right answer, and one that a
# table which cannot be matched by name is refused instead of being read
# positionally.
using DataFrames
using Tables

"""
Newsvendor with a single scalar [`Forecast`](@ref) named `d`, and a perfect
identity forecast model, so that the assessed cost of a sample is
`(c - q) * d = -4d`.
"""
function _inputs_newsvendor(; kwargs...)
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
    ADL.set_forecast_model(
        m,
        Chain(Dense(1 => 1; bias = false, init = (s...) -> ones(s...)));
        kwargs...,
    )
    return m, d
end

"""
Absolute-deviation model with a single scalar [`Forecast`](@ref) predicted from
**two** input features carrying different weights, so that reading the input
columns in the wrong order changes the cost. The assessed cost of a sample is
`|ŷ - y|`.
"""
function _two_feature_model(; kwargs...)
    m = ADL.Model()
    @variables(m, begin
        x, ADL.Policy
        d, ADL.Forecast
    end)
    @constraint(ADL.Plan(m), x.plan >= d.plan)
    @objective(ADL.Plan(m), Min, x.plan)
    @variable(ADL.Assess(m), s >= 0)
    @constraints(ADL.Assess(m), begin
        s >= d.assess - x.assess
        s >= x.assess - d.assess
    end)
    @objective(ADL.Assess(m), Min, s)
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    ADL.set_forecast_model(
        m,
        Chain(Dense(2 => 1; bias = false, init = (s...) -> [1.0 10.0]));
        kwargs...,
    )
    return m, d
end

"""
Absolute-deviation model with a single scalar [`Forecast`](@ref) and a single
input feature read by an identity network, so the assessed cost of a sample is
`|x - y|`. Zero cost is then proof that a realized value met the input row it
belongs to, which is what makes row matching observable.
"""
function _absdev_model(; kwargs...)
    m = ADL.Model()
    @variables(m, begin
        x, ADL.Policy
        d, ADL.Forecast
    end)
    @constraint(ADL.Plan(m), x.plan >= d.plan)
    @objective(ADL.Plan(m), Min, x.plan)
    @variable(ADL.Assess(m), s >= 0)
    @constraints(ADL.Assess(m), begin
        s >= d.assess - x.assess
        s >= x.assess - d.assess
    end)
    @objective(ADL.Assess(m), Min, s)
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    ADL.set_forecast_model(
        m,
        Chain(Dense(1 => 1; bias = false, init = (s...) -> ones(s...)));
        kwargs...,
    )
    return m, d
end

"""
Two [`Forecast`](@ref) variables declared as a container, so they are named
`f[1]` and `f[2]` rather than after anything a table would hold. The
coefficients differ, so swapping the two series changes the cost.
"""
function _container_forecast_model(; kwargs...)
    m = ADL.Model()
    @variable(m, x, ADL.Policy)
    @variable(m, f[1:2], ADL.Forecast)
    @constraint(ADL.Plan(m), x.plan >= f[1].plan + 2f[2].plan)
    @objective(ADL.Plan(m), Min, x.plan)
    @variable(ADL.Assess(m), s >= 0)
    @constraint(ADL.Assess(m), s >= f[1].assess + 2f[2].assess - x.assess)
    @objective(ADL.Assess(m), Min, x.assess + s)
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    ADL.set_forecast_model(
        m,
        Chain(Dense(1 => 2; bias = false, init = (s...) -> ones(s...)));
        kwargs...,
    )
    return m, f
end

"""
Two scalar [`Forecast`](@ref) variables named `d` and `e`, so that matching
table columns by name can be told apart from matching them by position.
"""
function _two_forecast_model()
    m = ADL.Model()
    @variables(m, begin
        x, ADL.Policy
        d, ADL.Forecast
        e, ADL.Forecast
    end)
    # the coefficients differ, so swapping d and e changes the cost
    @constraint(ADL.Plan(m), x.plan >= d.plan + 2e.plan)
    @objective(ADL.Plan(m), Min, x.plan)
    @variable(ADL.Assess(m), s >= 0)
    @constraint(ADL.Assess(m), s >= d.assess + 2e.assess - x.assess)
    @objective(ADL.Assess(m), Min, x.assess + s)
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    ADL.set_forecast_model(
        m,
        Chain(Dense(1 => 2; bias = false, init = (s...) -> ones(s...))),
    )
    return m, d, e
end

@testset "_forecast_base_name" begin
    m = ADL.Model()
    @variable(m, d, ADL.Forecast)
    @variable(m, f[1:2], ADL.Forecast)
    @test ADL._forecast_base_name(d) == "d"
    @test ADL._forecast_base_name(f[1]) == "f[1]"
    @test ADL._forecast_base_name(f[2]) == "f[2]"

    # an anonymous forecast variable has no name to match a column against
    info = JuMP.VariableInfo(
        false,
        0.0,
        false,
        0.0,
        false,
        0.0,
        false,
        0.0,
        false,
        false,
    )
    anon =
        JuMP.add_variable(m, JuMP.build_variable(error, info, ADL.Forecast), "")
    @test ADL._forecast_base_name(anon) == ""
end

@testset "compute_cost accepts every input container" begin
    Xm = reshape([10.0, 20.0], 2, 1)
    Yv = [10.0, 20.0]

    # `compute_cost` does not mutate the predictive model, so one model serves
    # every container - building a fresh one per case dominated the runtime
    # every table `X` below holds its single feature in a column called `a`
    m, d = _inputs_newsvendor(; input_names = [:a])

    # the `Dict` form is the reference: it is keyed by the variables themselves
    expected = ADL.compute_cost(m, Xm, Dict(d => Yv), false, false)
    @test expected ≈ [-40.0, -80.0] atol = 1e-6

    equivalent = [
        ("matrix, matrix", Xm, reshape(Yv, 2, 1)),
        ("matrix, vector", Xm, Yv),
        ("vector, vector", [10.0, 20.0], Yv),
        ("namedtuple, namedtuple", (a = [10.0, 20.0],), (d = Yv,)),
        (
            "dataframe, dataframe",
            DataFrame(a = [10.0, 20.0]),
            DataFrame(d = Yv),
        ),
        # a column that is not a forecast variable is ignored when the forecast
        # names are all present
        (
            "dataframe with extra column",
            DataFrame(a = [10.0, 20.0]),
            DataFrame(date = [1, 2], d = Yv),
        ),
        # mixed containers across the two arguments
        ("dataframe, vector", DataFrame(a = [10.0, 20.0]), Yv),
        ("dataframe, dict", DataFrame(a = [10.0, 20.0]), Dict(d => Yv)),
        # integer columns are `Real`, as an Int matrix already is
        ("integer columns", DataFrame(a = [10, 20]), DataFrame(d = [10, 20])),
    ]
    for (label, X, Y) in equivalent
        @test ADL.compute_cost(m, X, Y, false, false) ≈ expected atol = 1e-6
    end
end

@testset "table columns are matched to forecast variables by name" begin
    X = reshape([1.0, 2.0], 2, 1)
    m, d, e = _two_forecast_model()
    @test [ADL._forecast_base_name(f) for f in m.forecast.output_variables] == ["d", "e"]
    expected = ADL.compute_cost(
        m,
        X,
        Dict(d => [3.0, 4.0], e => [5.0, 6.0]),
        false,
        false,
    )

    # columns deliberately in the wrong order: matching by name must recover the
    # right pairing, where matching by position would silently swap the series
    m2, _, _ = _two_forecast_model()
    swapped = DataFrame(e = [5.0, 6.0], d = [3.0, 4.0])
    @test ADL.compute_cost(m2, X, swapped, false, false) ≈ expected atol = 1e-6

    # and the swap really is observable, i.e. this model can tell them apart.
    # `Matrix` is also the documented way to ask for positional matching: it
    # drops the names, which is exactly the request being made
    m3, _, _ = _two_forecast_model()
    positional = ADL.compute_cost(
        m3,
        X,
        Matrix(DataFrame(first = [5.0, 6.0], second = [3.0, 4.0])),
        false,
        false,
    )
    @test !(positional ≈ expected)

    # a table that cannot be matched by name is refused rather than read
    # positionally, even though it has exactly the right number of columns
    m4, _, _ = _two_forecast_model()
    @test_throws ArgumentError ADL.compute_cost(
        m4,
        X,
        DataFrame(first = [5.0, 6.0], second = [3.0, 4.0]),
        false,
        false,
    )
end

@testset "a single column is no exception" begin
    # A one-column table's order cannot be wrong, but its name still can: a
    # one-input model handed the wrong column is a mistake worth catching, and only
    # a declared schema catches it. So narrow tables get no carve-out.
    Xm = reshape([10.0, 20.0], 2, 1)
    Yv = [10.0, 20.0]

    bare, d = _inputs_newsvendor()
    @test isnothing(bare.forecast.input_names)
    @test bare.forecast.input_size == 1
    reference = ADL.compute_cost(bare, Xm, Dict(d => Yv), false, false)
    # a matrix is unaffected: it never reaches the column matching at all
    @test ADL.compute_cost(bare, Xm, Yv, false, false) ≈ reference atol = 1e-6
    # a one-column table is not
    @test_throws ArgumentError ADL.compute_cost(
        bare,
        DataFrame(a = [10.0, 20.0]),
        Yv,
        false,
        false,
    )

    # declaring the name is the fix, and it then refuses the wrong column
    named, _ = _inputs_newsvendor(; input_names = [:a])
    @test ADL.compute_cost(
        named,
        DataFrame(a = [10.0, 20.0]),
        Yv,
        false,
        false,
    ) ≈ reference atol = 1e-6
    @test_throws ArgumentError ADL.compute_cost(
        named,
        DataFrame(humidity = [10.0, 20.0]),
        Yv,
        false,
        false,
    )

    # same on the output side, where a scalar forecast variable already names its
    # own column, so a differently named one is refused rather than taken by order
    @test ADL.compute_cost(
        named,
        DataFrame(a = [10.0, 20.0]),
        DataFrame(d = Yv),
        false,
        false,
    ) ≈ reference atol = 1e-6
    @test_throws ArgumentError ADL.compute_cost(
        named,
        DataFrame(a = [10.0, 20.0]),
        DataFrame(zzz = Yv),
        false,
        false,
    )
end

@testset "X columns are selected by input_names" begin
    X = [1.0 2.0; 3.0 4.0]
    Y = [21.0, 43.0]  # exactly the prediction of the correctly ordered input

    # with no declared schema a multi-column table cannot be read at all: there
    # is nothing to match its names against, and its order is not to be trusted
    plain, _ = _two_feature_model()
    @test isnothing(plain.forecast.input_names)
    @test ADL.compute_cost(plain, X, Y, false, false) ≈ [0.0, 0.0] atol = 1e-6
    @test_throws ArgumentError ADL.compute_cost(
        plain,
        DataFrame(a = X[:, 1], b = X[:, 2]),
        Y,
        false,
        false,
    )
    # ... and the mistake it is protecting against is a real one: this model
    # reads its two features with different weights
    @test !isapprox(
        ADL.compute_cost(plain, X[:, [2, 1]], Y, false, false),
        [0.0, 0.0];
        atol = 1e-6,
    )

    named, _ = _two_feature_model(; input_names = [:a, :b])
    @test named.forecast.input_names == [:a, :b]
    ordered = DataFrame(a = X[:, 1], b = X[:, 2])
    @test ADL.compute_cost(named, ordered, Y, false, false) ≈ [0.0, 0.0] atol =
        1e-6
    # the whole point: the same data with the columns the other way round, and
    # with an unrelated column in the way, reaches the same answer
    @test ADL.compute_cost(
        named,
        DataFrame(b = X[:, 2], a = X[:, 1]),
        Y,
        false,
        false,
    ) ≈ [0.0, 0.0] atol = 1e-6
    @test ADL.compute_cost(
        named,
        DataFrame(id = [1, 2], b = X[:, 2], a = X[:, 1]),
        Y,
        false,
        false,
    ) ≈ [0.0, 0.0] atol = 1e-6
    # a declared column that is not there is named in the error
    @test_throws ArgumentError ADL.compute_cost(
        named,
        DataFrame(a = X[:, 1], zz = X[:, 2]),
        Y,
        false,
        false,
    )
    # a matrix is still positional, so a declared schema does not change it
    @test ADL.compute_cost(named, X, Y, false, false) ≈ [0.0, 0.0] atol = 1e-6
end

@testset "Symbol keys in the input_output_map declare the schema" begin
    m = ADL.Model()
    @variables(m, begin
        x, ADL.Policy
        d, ADL.Forecast
        e, ADL.Forecast
    end)
    nets = [Dense(1 => 1; bias = false), Dense(1 => 1; bias = false)]
    named_map = [Dict([:zz] => [d]), Dict([:aa] => [e])]

    pm = ADL.PredictiveModel(nets, named_map)
    # the schema defaults to the names the map uses, in alphabetical order -
    # `keys(::Dict)` has no defined order, so it cannot be first-appearance
    @test pm.input_names == [:aa, :zz]
    @test pm.input_size == 2
    # resolved to positions in that order: `d` reads `zz`, column 2
    @test pm.input_output_map[1] == Dict([2] => [d])
    @test pm.input_output_map[2] == Dict([1] => [e])

    # an explicit `input_names` fixes a different column order instead
    pm2 = ADL.PredictiveModel(nets, named_map; input_names = [:zz, :aa])
    @test pm2.input_names == [:zz, :aa]
    @test pm2.input_output_map[1] == Dict([1] => [d])
    @test pm2.input_output_map[2] == Dict([2] => [e])

    # and it has to cover every name the map refers to
    @test_throws ArgumentError ADL.PredictiveModel(
        nets,
        named_map;
        input_names = [:zz, :bb],
    )

    # the names survive `set_forecast_model`, which rebuilds the model
    ADL.set_forecast_model(m, pm)
    @test m.forecast.input_names == [:aa, :zz]
end

@testset "output_names names the columns of container forecasts" begin
    # `@variable(m, f[1:2], Forecast)` names its variables `f[1]` and `f[2]`, which
    # no table is going to carry, so without `output_names` those models would be
    # stuck with the `Dict` form
    m, f = _container_forecast_model(; output_names = [:demand, :price])
    @test m.forecast.output_names == [:demand, :price]
    X = reshape([1.0, 2.0], 2, 1)
    expected = ADL.compute_cost(
        m,
        X,
        Dict(f[1] => [3.0, 4.0], f[2] => [5.0, 6.0]),
        false,
        false,
    )
    @test ADL.compute_cost(
        m,
        X,
        DataFrame(price = [5.0, 6.0], demand = [3.0, 4.0]),
        false,
        false,
    ) ≈ expected atol = 1e-6

    # without it, the variables' own names are looked for and not found
    bare, _ = _container_forecast_model()
    @test isnothing(bare.forecast.output_names)
    @test ADL._output_column_names(bare.forecast) ==
          [Symbol("f[1]"), Symbol("f[2]")]
    @test_throws ArgumentError ADL.compute_cost(
        bare,
        X,
        DataFrame(demand = [3.0, 4.0], price = [5.0, 6.0]),
        false,
        false,
    )
end

@testset "sample_key matches rows instead of trusting their order" begin
    # the forecast is the identity and the cost is |ŷ - y|, so a zero cost for
    # every sample is proof that each realized value met the right input row
    X = DataFrame(t = [10, 20, 30], a = [1.0, 2.0, 3.0])
    aligned = DataFrame(t = [10, 20, 30], d = [1.0, 2.0, 3.0])
    shuffled = DataFrame(t = [30, 10, 20], d = [3.0, 1.0, 2.0])

    keyed, _ = _absdev_model(; input_names = [:a], sample_key = :t)
    @test keyed.forecast.sample_key == :t
    @test ADL.compute_cost(keyed, X, aligned, false, false) ≈ zeros(3) atol =
        1e-6
    @test ADL.compute_cost(keyed, X, shuffled, false, false) ≈ zeros(3) atol =
        1e-6

    # and the mistake it protects against is a real one: without the key the very
    # same shuffled table is read in order, and quietly gives a different answer
    bare, _ = _absdev_model(; input_names = [:a])
    @test isnothing(bare.forecast.sample_key)
    @test ADL.compute_cost(bare, X, aligned, false, false) ≈ zeros(3) atol =
        1e-6
    @test !isapprox(
        ADL.compute_cost(bare, X, shuffled, false, false),
        zeros(3);
        atol = 1e-6,
    )

    # `Y` is looked up per row of `X`, so it may cover samples `X` does not ask
    # for - the row-wise counterpart of ignoring an unwanted column
    superset =
        DataFrame(t = [99, 30, 10, 20, 77], d = [-1.0, 3.0, 1.0, 2.0, -1.0])
    @test ADL.compute_cost(keyed, X, superset, false, false) ≈ zeros(3) atol =
        1e-6

    # the key column identifies a row, it is not a feature: it is neither fed to
    # the network nor counted among the columns the caller is asked to name
    @test keyed.forecast.input_size == 1
    msg = try
        ADL._matched_columns([:t, :a], nothing, 1, :t, "Input data `X`", "hint")
        ""
    catch e
        e.msg
    end
    @test occursin("1 matchable column", msg)
    @test occursin("(a)", msg)

    # a container that carries no row labels falls back to order, exactly as an
    # unnamed container falls back to position on the column side
    Xm = reshape([1.0, 2.0, 3.0], 3, 1)
    @test ADL.compute_cost(keyed, Xm, aligned, false, false) ≈ zeros(3) atol =
        1e-6
    d = keyed.forecast.output_variables[1]
    @test ADL.compute_cost(keyed, X, Dict(d => [1.0, 2.0, 3.0]), false, false) ≈
          zeros(3) atol = 1e-6
end

@testset "sample_key errors" begin
    X = DataFrame(t = [10, 20, 30], a = [1.0, 2.0, 3.0])
    keyed, _ = _absdev_model(; input_names = [:a], sample_key = :t)

    # a sample `X` asks for that `Y` does not have is an error, not an off-by-one
    @test_throws ArgumentError ADL.compute_cost(
        keyed,
        X,
        DataFrame(t = [10, 20, 99], d = [1.0, 2.0, 3.0]),
        false,
        false,
    )
    # a key that repeats is not a key: the pairing would be ambiguous
    @test_throws ArgumentError ADL.compute_cost(
        keyed,
        X,
        DataFrame(t = [10, 20, 20], d = [1.0, 2.0, 3.0]),
        false,
        false,
    )
    @test_throws ArgumentError ADL.compute_cost(
        keyed,
        DataFrame(t = [10, 10, 30], a = [1.0, 2.0, 3.0]),
        DataFrame(t = [30, 10, 20], d = [3.0, 1.0, 2.0]),
        false,
        false,
    )
    # declaring a key asserts that the tables carry it
    @test_throws ArgumentError ADL.compute_cost(
        keyed,
        X,
        DataFrame(d = [1.0, 2.0, 3.0]),
        false,
        false,
    )
    @test_throws ArgumentError ADL.compute_cost(
        keyed,
        DataFrame(a = [1.0, 2.0, 3.0]),
        DataFrame(t = [10, 20, 30], d = [1.0, 2.0, 3.0]),
        false,
        false,
    )
end

@testset "train! aligns rows by sample_key" begin
    # end to end, since the alignment happens once at the entry point rather than
    # inside the epoch loop: training from a shuffled table has to follow the same
    # trajectory as training from the aligned one
    X = DataFrame(t = [10, 20, 30], a = [1.0, 2.0, 3.0])
    aligned = DataFrame(t = [10, 20, 30], d = [1.0, 2.0, 3.0])
    shuffled = DataFrame(t = [30, 10, 20], d = [3.0, 1.0, 2.0])
    opt = ADL.Options(
        ADL.GradientMode;
        rule = Flux.Adam(0.1),
        epochs = 3,
        verbose = false,
    )

    ms, _ = _absdev_model(; input_names = [:a], sample_key = :t)
    ma, _ = _absdev_model(; input_names = [:a], sample_key = :t)
    sol_shuffled = ADL.train!(ms, X, shuffled, opt)
    sol_aligned = ADL.train!(ma, X, aligned, opt)
    @test sol_shuffled.cost ≈ sol_aligned.cost atol = 1e-6
    @test sol_shuffled.params ≈ sol_aligned.params atol = 1e-6
end

@testset "declared names are validated against the model" begin
    @test_throws ArgumentError _two_feature_model(; input_names = [:a])
    @test_throws ArgumentError _two_feature_model(; input_names = [:a, :b, :c])
    # a repeated name would feed two slots from the same column
    @test_throws ArgumentError _two_feature_model(; input_names = [:a, :a])
    @test_throws ArgumentError _inputs_newsvendor(; output_names = [:a, :b])
    @test_throws ArgumentError _container_forecast_model(;
        output_names = [:same, :same],
    )
end

@testset "shape mismatches are reported as ArgumentError" begin
    # `@assert` is documented as removable at some optimization levels, so these
    # data checks must not be assertions
    m, _ = _inputs_newsvendor()
    Xm = reshape([10.0, 20.0], 2, 1)
    @test_throws ArgumentError ADL.compute_cost(
        m,
        Xm,
        [1.0, 2.0, 3.0],
        false,
        false,
    )
    @test_throws ArgumentError ADL.compute_cost(
        m,
        [1.0 2.0; 3.0 4.0],
        [1.0, 2.0],
        false,
        false,
    )
    @test_throws ArgumentError ADL.compute_cost(
        m,
        Xm,
        [1.0 2.0; 3.0 4.0],
        false,
        false,
    )
end

@testset "train! accepts tables" begin
    # end to end: training from a DataFrame must follow the same trajectory as
    # training from the equivalent Dict. Only a few epochs are needed - the point
    # is that the paths agree, not that they converge.
    Xd = DataFrame(a = ones(4))
    Yd = DataFrame(d = fill(50.0, 4))
    opt = ADL.Options(
        ADL.GradientMode;
        rule = Flux.Adam(1.0),
        epochs = 3,
        verbose = false,
    )

    mt, _ = _inputs_newsvendor(; input_names = [:a])
    sol_table = ADL.train!(mt, Xd, Yd, opt)

    md, dd = _inputs_newsvendor()
    sol_dict = ADL.train!(md, ones(4, 1), Dict(dd => fill(50.0, 4)), opt)

    @test sol_table.cost ≈ sol_dict.cost atol = 1e-6
    @test sol_table.params ≈ sol_dict.params atol = 1e-6

    # a NamedTuple table and a vector reach the same place
    mn, _ = _inputs_newsvendor(; input_names = [:a])
    sol_nt = ADL.train!(mn, (a = ones(4),), (d = fill(50.0, 4),), opt)
    @test sol_nt.cost ≈ sol_dict.cost atol = 1e-6
end

@testset "input container errors" begin
    m, _ = _inputs_newsvendor(; input_names = [:a])
    Xm = reshape([10.0, 20.0], 2, 1)
    Yv = [10.0, 20.0]

    # non-numeric and missing data must be reported against the argument, not
    # deep inside Flux or the solver
    @test_throws ArgumentError ADL.compute_cost(
        m,
        Xm,
        DataFrame(d = ["a", "b"]),
    )
    @test_throws ArgumentError ADL.compute_cost(
        m,
        Xm,
        DataFrame(d = [1.0, missing]),
    )
    @test_throws ArgumentError ADL.compute_cost(
        m,
        DataFrame(a = ["x", "y"]),
        Yv,
    )

    # right shape, but neither name-matched nor the right number of columns
    @test_throws ArgumentError ADL.compute_cost(
        m,
        Xm,
        DataFrame(p = Yv, q = Yv),
    )

    # wholly unsupported containers
    @test_throws ArgumentError ADL.compute_cost(m, Xm, "nonsense")
    @test_throws ArgumentError ADL.compute_cost(m, "nonsense", Yv)

    # the forecast model has to be set before inputs can be interpreted at all,
    # since reading `Y` depends on the forecast variables
    m2 = ADL.Model()
    @variables(m2, begin
        x, ADL.Policy
        d, ADL.Forecast
    end)
    @test_throws ArgumentError ADL.compute_cost(
        m2,
        DataFrame(a = [1.0]),
        DataFrame(d = [1.0]),
    )
    @test_throws ArgumentError ADL.train!(
        m2,
        DataFrame(a = [1.0]),
        DataFrame(d = [1.0]),
        ADL.Options(ADL.GradientMode; epochs = 1),
    )
end

@testset "table entry points are not over-specialized" begin
    # The generic `compute_cost` / `train!` wrappers take their containers as
    # `@nospecialize`. Without that, specializing a wrapper on a concrete
    # container type drags the entire `compute_cost` body through inference again
    # in the new context - measured at 174s for the first table-typed call
    # against 0.001s once compiled, which is what this guards against.
    #
    # `Base.specializations` reports what the compiler actually instantiated, so
    # a single entry for a method called with several unrelated container types
    # is direct evidence the annotation is still doing its job.
    m, d = _inputs_newsvendor(; input_names = [:a])
    Yv = [10.0, 20.0]
    container_pairs = [
        ((a = Yv,), (d = Yv,)),
        ((a = Yv,), DataFrame(d = Yv)),
        (DataFrame(a = Yv), DataFrame(d = Yv)),
        (Yv, Yv),
        ((a = Yv,), Yv),
        (DataFrame(a = Yv), Dict(d => Yv)),
    ]
    for (Xa, Ya) in container_pairs
        ADL.compute_cost(m, Xa, Ya, false, false)
    end

    generic = only(
        methods(
            ADL.compute_cost,
            Tuple{ADL.Model,NamedTuple,NamedTuple,Bool,Bool},
        ),
    )
    signatures = [string(s.specTypes) for s in Base.specializations(generic)]

    # the sharp statement of intent: nothing the compiler instantiated mentions a
    # concrete container type, so the wrapper is not specialized per container
    @test !any(s -> occursin("NamedTuple", s), signatures)
    @test !any(s -> occursin("DataFrame", s), signatures)
    # and the count stays well under one-per-container-type
    @test length(signatures) < length(container_pairs)
end

@testset "series length mismatch inside a table" begin
    # a table cannot normally hold ragged columns, but the column-wise Tables.jl
    # interface can be satisfied by a plain `Dict` of vectors
    m, _ = _two_forecast_model()
    ragged = Dict(:d => [1.0, 2.0], :e => [1.0])
    @test_throws ArgumentError ADL.compute_cost(
        m,
        reshape([1.0, 2.0], 2, 1),
        ragged,
    )
end
