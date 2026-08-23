# Worked examples of linking data to a model, one per way of doing it.
#
# Every testset here is DELIBERATELY self-contained: no shared helper builds the
# model, the numbers are written out, and the arithmetic is in the comments. The
# point is that a reader can open any one of them and see the whole picture -
# which policy variable, which forecast variable, which column of which container
# - without chasing a constructor somewhere else in the file. The duplication is
# the feature. `test_data_inputs.jl` is the opposite trade: helpers and edge cases.
#
# All six spell out the SAME problem and must produce the SAME cost,
# `[300.0, 450.0]`, so the differences between them are purely about how data is
# handed over.
#
# ## The problem, once
#
# Two generators, each serving its own load, with different costs and different
# shortfall penalties - asymmetric on purpose, so pairing the wrong series with
# the wrong variable changes the answer and cannot pass unnoticed.
#
#   policy   : g1, g2          generation committed ahead of time
#   forecast : load_a, load_b  the two loads, predicted from weather
#
#   plan   : min 10*g1 + 20*g2          s.t. g1 >= load_a, g2 >= load_b
#            => g1 = ŷ_a, g2 = ŷ_b
#   assess : min 10*g1 + 20*g2 + 50*s1 + 30*s2
#            s.t. s1 >= load_a - g1, s2 >= load_b - g2   (g1, g2 already fixed)
#            => cost = 10ŷ_a + 20ŷ_b + 50*max(0, y_a - ŷ_a) + 30*max(0, y_b - ŷ_b)
#
# The forecast model is the 2x2 identity with no bias, so ŷ_a is literally the
# first input and ŷ_b the second. Nothing is hidden in the network.
#
#   sample 1: temp=10 wind= 5 | y_a=12 y_b= 4
#             100 + 100 + 50*max(0,  2) + 30*max(0, -1) = 300
#   sample 2: temp=20 wind= 8 | y_a=18 y_b=11
#             200 + 160 + 50*max(0, -2) + 30*max(0,  3) = 450
using DataFrames

const _LINK_COST = [300.0, 450.0]

# the identity, spelled out, so the prediction is visibly the input
_identity_2x2(s...) = [1.0 0.0; 0.0 1.0]
_identity_1x1(s...) = ones(s...)

@testset "1. matrices: everything by position" begin
    # Policy and Forecast variables declared with separate `@variable` calls, which
    # is equivalent to one `@variables` block: each call appends to the model, and
    # the resulting DECLARATION ORDER is what a positional container refers to.
    model = ADL.Model()
    @variable(model, g1 >= 0, ADL.Policy)
    @variable(model, g2 >= 0, ADL.Policy)
    @variable(model, load_a, ADL.Forecast)
    @variable(model, load_b, ADL.Forecast)

    @constraints(ADL.Plan(model), begin
        g1.plan >= load_a.plan
        g2.plan >= load_b.plan
    end)
    @objective(ADL.Plan(model), Min, 10 * g1.plan + 20 * g2.plan)

    @variables(ADL.Assess(model), begin
        s1 >= 0
        s2 >= 0
    end)
    @constraints(ADL.Assess(model), begin
        s1 >= load_a.assess - g1.assess
        s2 >= load_b.assess - g2.assess
    end)
    @objective(
        ADL.Assess(model),
        Min,
        10 * g1.assess + 20 * g2.assess + 50 * s1 + 30 * s2
    )

    set_optimizer(model, HiGHS.Optimizer)
    set_silent(model)
    ADL.set_forecast_model(
        model,
        ADL.ForecastModel(
            architecture = Chain(
                Dense(2 => 2; bias = false, init = _identity_2x2),
            ),
            outputs = [load_a, load_b],
        ),
    )

    # two Policy, two Forecast, in declaration order
    @test length(model.policy_vars) == 2
    @test [ADL._forecast_base_name(f) for f in model.forecast_vars] == ["load_a", "load_b"]

    # X column 1 is the network's first input, column 2 its second
    X = [
        10.0 5.0    # temp, wind for sample 1
        20.0 8.0
    ]   # ... and sample 2
    # Y column 1 is load_a, column 2 is load_b - that is, forecast declaration
    # order. Nothing here says so; variation 6 is this same matrix with the order
    # written down, and the named containers below say it a different way.
    Y = [
        12.0 4.0
        18.0 11.0
    ]

    @test ADL.compute_cost(model, X, Y, false, false) ≈ _LINK_COST atol = 1e-6
    # aggregate=true averages over the samples
    @test ADL.compute_cost(model, X, Y) ≈ sum(_LINK_COST) / 2 atol = 1e-6

    # getting it wrong is silent: swapping the two Y columns still runs, and just
    # returns a different number
    @test !isapprox(
        ADL.compute_cost(model, X, Y[:, [2, 1]], false, false),
        _LINK_COST;
        atol = 1e-6,
    )
end

@testset "2. matrix X, Dict Y: realized values keyed by the variables" begin
    model = ADL.Model()
    @variables(model, begin
        g1 >= 0, ADL.Policy
        g2 >= 0, ADL.Policy
        load_a, ADL.Forecast
        load_b, ADL.Forecast
    end)

    @constraints(ADL.Plan(model), begin
        g1.plan >= load_a.plan
        g2.plan >= load_b.plan
    end)
    @objective(ADL.Plan(model), Min, 10 * g1.plan + 20 * g2.plan)

    @variables(ADL.Assess(model), begin
        s1 >= 0
        s2 >= 0
    end)
    @constraints(ADL.Assess(model), begin
        s1 >= load_a.assess - g1.assess
        s2 >= load_b.assess - g2.assess
    end)
    @objective(
        ADL.Assess(model),
        Min,
        10 * g1.assess + 20 * g2.assess + 50 * s1 + 30 * s2
    )

    set_optimizer(model, HiGHS.Optimizer)
    set_silent(model)
    ADL.set_forecast_model(
        model,
        ADL.ForecastModel(
            architecture = Chain(
                Dense(2 => 2; bias = false, init = _identity_2x2),
            ),
            outputs = [load_a, load_b],
        ),
    )

    X = [
        10.0 5.0
        20.0 8.0
    ]
    # keyed by the Forecast variables themselves, so declaration order is
    # irrelevant and there is nothing to mispair. This is the most explicit form
    # available, and the reference the other variations are checked against.
    Y = Dict(load_b => [4.0, 11.0], load_a => [12.0, 18.0])

    @test ADL.compute_cost(model, X, Y, false, false) ≈ _LINK_COST atol = 1e-6

    # a variable with no series is named, rather than surfacing as a `KeyError`
    @test_throws ArgumentError ADL.compute_cost(
        model,
        X,
        Dict(load_a => [12.0, 18.0]),
        false,
        false,
    )
end

@testset "3. tables: X by the unit's input names, Y by the variable names" begin
    model = ADL.Model()
    @variable(model, g1 >= 0, ADL.Policy)
    @variable(model, g2 >= 0, ADL.Policy)
    @variable(model, load_a, ADL.Forecast)
    @variable(model, load_b, ADL.Forecast)

    @constraints(ADL.Plan(model), begin
        g1.plan >= load_a.plan
        g2.plan >= load_b.plan
    end)
    @objective(ADL.Plan(model), Min, 10 * g1.plan + 20 * g2.plan)

    @variables(ADL.Assess(model), begin
        s1 >= 0
        s2 >= 0
    end)
    @constraints(ADL.Assess(model), begin
        s1 >= load_a.assess - g1.assess
        s2 >= load_b.assess - g2.assess
    end)
    @objective(
        ADL.Assess(model),
        Min,
        10 * g1.assess + 20 * g2.assess + 50 * s1 + 30 * s2
    )

    set_optimizer(model, HiGHS.Optimizer)
    set_silent(model)
    # naming the unit's `inputs` is what declares the input schema: the network's
    # first input is the `temp` column, its second the `wind` column. `Y` needs
    # nothing, since `load_a` and `load_b` already name their own columns.
    ADL.set_forecast_model(
        model,
        ADL.ForecastModel(
            inputs = [:temp, :wind],
            architecture = Chain(
                Dense(2 => 2; bias = false, init = _identity_2x2),
            ),
            outputs = [load_a, load_b],
        ),
    )
    @test model.forecast.input_names == [:temp, :wind]
    @test isnothing(model.forecast.output_columns)

    # columns deliberately in the "wrong" order on both sides, plus a stray column
    X = DataFrame(wind = [5.0, 8.0], station = ["A", "B"], temp = [10.0, 20.0])
    Y = DataFrame(load_b = [4.0, 11.0], load_a = [12.0, 18.0])
    @test ADL.compute_cost(model, X, Y, false, false) ≈ _LINK_COST atol = 1e-6

    # a `station` column of strings is fine as long as it is not asked for; only
    # the declared columns are read
    @test ADL.compute_cost(
        model,
        DataFrame(temp = [10.0, 20.0], wind = [5.0, 8.0]),
        Y,
        false,
        false,
    ) ≈ _LINK_COST atol = 1e-6

    # and the mistake variation 1 could not detect is now an error, not a number
    @test_throws ArgumentError ADL.compute_cost(
        model,
        DataFrame(temperature = [10.0, 20.0], wind = [5.0, 8.0]),
        Y,
        false,
        false,
    )
    @test_throws ArgumentError ADL.compute_cost(
        model,
        X,
        DataFrame(load_a = [12.0, 18.0], load_c = [4.0, 11.0]),
        false,
        false,
    )

    # `Matrix` drops the names and is how you ask for variation 1's reading back
    @test ADL.compute_cost(
        model,
        Matrix(DataFrame(temp = [10.0, 20.0], wind = [5.0, 8.0])),
        Matrix(Y[:, [:load_a, :load_b]]),
        false,
        false,
    ) ≈ _LINK_COST atol = 1e-6
end

@testset "4. one network per forecast, each naming its own Y column" begin
    # Same problem, but the loads are declared as a CONTAINER - so they are named
    # `load[1]` and `load[2]`, which no data file is going to carry - and each is
    # predicted by its own network from its own column.
    model = ADL.Model()
    @variable(model, g[1:2] >= 0, ADL.Policy)
    @variable(model, load[1:2], ADL.Forecast)

    @constraints(ADL.Plan(model), begin
        g[1].plan >= load[1].plan
        g[2].plan >= load[2].plan
    end)
    @objective(ADL.Plan(model), Min, 10 * g[1].plan + 20 * g[2].plan)

    @variables(ADL.Assess(model), begin
        s1 >= 0
        s2 >= 0
    end)
    @constraints(ADL.Assess(model), begin
        s1 >= load[1].assess - g[1].assess
        s2 >= load[2].assess - g[2].assess
    end)
    @objective(
        ADL.Assess(model),
        Min,
        10 * g[1].assess + 20 * g[2].assess + 50 * s1 + 30 * s2
    )

    set_optimizer(model, HiGHS.Optimizer)
    set_silent(model)

    # each unit says what it reads and what it predicts: unit 1 reads `temp` and
    # predicts load[1], unit 2 reads `wind` and predicts load[2]. The names are
    # resolved to column positions when the units are assembled.
    predictive = ADL.FullForecastModel([
        ADL.ForecastModel(
            inputs = [:temp],
            architecture = Dense(1 => 1; bias = false, init = _identity_1x1),
            # the variables are called `load[1]` / `load[2]`, so name the Y column
            outputs = [load[1] => :load_a],
        ),
        ADL.ForecastModel(
            inputs = [:wind],
            architecture = Dense(1 => 1; bias = false, init = _identity_1x1),
            outputs = [load[2] => :load_b],
        ),
    ])
    # derived from the units, in order of first appearance
    @test predictive.input_names == [:temp, :wind]
    @test predictive.units[1].inputs == [1]
    @test predictive.units[1].outputs == [load[1]]
    @test predictive.units[2].inputs == [2]
    @test predictive.units[2].outputs == [load[2]]

    ADL.set_forecast_model(model, predictive)
    @test [
        ADL._forecast_base_name(f) for f in model.forecast.output_variables
    ] == ["load[1]", "load[2]"]
    @test model.forecast.output_columns == [:load_a, :load_b]

    X = DataFrame(wind = [5.0, 8.0], temp = [10.0, 20.0])
    Y = DataFrame(load_b = [4.0, 11.0], load_a = [12.0, 18.0])
    @test ADL.compute_cost(model, X, Y, false, false) ≈ _LINK_COST atol = 1e-6

    # two networks, one weight each
    @test length(ADL.extract_params(model.forecast)) == 2

    # gradients come back per sample, one column per forecast variable
    cost, dC = ADL.compute_cost(model, X, Y, true, false)
    @test cost ≈ _LINK_COST atol = 1e-6
    @test size(dC) == (2, 2)

    # and the same containers train
    solution = ADL.train!(
        model,
        X,
        Y,
        ADL.Options(
            ADL.GradientMode;
            rule = Flux.Adam(0.05),
            epochs = 2,
            verbose = false,
        ),
    )
    @test length(solution.params) == 2
    @test isfinite(solution.cost)
end

@testset "5. tables with a sample_key: rows matched too" begin
    model = ADL.Model()
    @variable(model, g1 >= 0, ADL.Policy)
    @variable(model, g2 >= 0, ADL.Policy)
    @variable(model, load_a, ADL.Forecast)
    @variable(model, load_b, ADL.Forecast)

    @constraints(ADL.Plan(model), begin
        g1.plan >= load_a.plan
        g2.plan >= load_b.plan
    end)
    @objective(ADL.Plan(model), Min, 10 * g1.plan + 20 * g2.plan)

    @variables(ADL.Assess(model), begin
        s1 >= 0
        s2 >= 0
    end)
    @constraints(ADL.Assess(model), begin
        s1 >= load_a.assess - g1.assess
        s2 >= load_b.assess - g2.assess
    end)
    @objective(
        ADL.Assess(model),
        Min,
        10 * g1.assess + 20 * g2.assess + 50 * s1 + 30 * s2
    )

    set_optimizer(model, HiGHS.Optimizer)
    set_silent(model)
    # `sample_key` names the column that identifies an observation. It is not a
    # feature: the network still takes 2 inputs, `temp` and `wind`.
    ADL.set_forecast_model(
        model,
        ADL.ForecastModel(
            inputs = [:temp, :wind],
            architecture = Chain(
                Dense(2 => 2; bias = false, init = _identity_2x2),
            ),
            outputs = [load_a, load_b],
        );
        sample_key = :hour,
    )
    @test model.forecast.input_size == 2
    @test model.forecast.sample_key == :hour

    X = DataFrame(hour = [1, 2], temp = [10.0, 20.0], wind = [5.0, 8.0])

    # the realized values in the same row order ...
    aligned =
        DataFrame(hour = [1, 2], load_a = [12.0, 18.0], load_b = [4.0, 11.0])
    @test ADL.compute_cost(model, X, aligned, false, false) ≈ _LINK_COST atol =
        1e-6

    # ... in the reverse row order, looked up by `hour` ...
    reversed =
        DataFrame(hour = [2, 1], load_a = [18.0, 12.0], load_b = [11.0, 4.0])
    @test ADL.compute_cost(model, X, reversed, false, false) ≈ _LINK_COST atol =
        1e-6

    # ... and covering hours `X` never asks about, which are simply not used
    superset = DataFrame(
        hour = [7, 2, 99, 1],
        load_a = [-1.0, 18.0, -1.0, 12.0],
        load_b = [-1.0, 11.0, -1.0, 4.0],
    )
    @test ADL.compute_cost(model, X, superset, false, false) ≈ _LINK_COST atol =
        1e-6

    # an hour `X` asks for and `Y` lacks is an error, not an off-by-one
    @test_throws ArgumentError ADL.compute_cost(
        model,
        X,
        DataFrame(hour = [1, 99], load_a = [12.0, 18.0], load_b = [4.0, 11.0]),
        false,
        false,
    )

    # without the key the reversed table would have been read in order, quietly
    # pairing hour 1's inputs with hour 2's realized values
    unkeyed = ADL.Model()
    @variable(unkeyed, g1u >= 0, ADL.Policy)
    @variable(unkeyed, g2u >= 0, ADL.Policy)
    @variable(unkeyed, load_a_u, ADL.Forecast)
    @variable(unkeyed, load_b_u, ADL.Forecast)
    @constraints(ADL.Plan(unkeyed), begin
        g1u.plan >= load_a_u.plan
        g2u.plan >= load_b_u.plan
    end)
    @objective(ADL.Plan(unkeyed), Min, 10 * g1u.plan + 20 * g2u.plan)
    @variables(ADL.Assess(unkeyed), begin
        s1u >= 0
        s2u >= 0
    end)
    @constraints(ADL.Assess(unkeyed), begin
        s1u >= load_a_u.assess - g1u.assess
        s2u >= load_b_u.assess - g2u.assess
    end)
    @objective(
        ADL.Assess(unkeyed),
        Min,
        10 * g1u.assess + 20 * g2u.assess + 50 * s1u + 30 * s2u
    )
    set_optimizer(unkeyed, HiGHS.Optimizer)
    set_silent(unkeyed)
    # the `Y` columns are named only because the variables had to be renamed to
    # avoid clashing with the ones above; it lets this model read the very same
    # tables
    ADL.set_forecast_model(
        unkeyed,
        ADL.ForecastModel(
            inputs = [:temp, :wind],
            architecture = Chain(
                Dense(2 => 2; bias = false, init = _identity_2x2),
            ),
            outputs = [load_a_u => :load_a, load_b_u => :load_b],
        ),
    )
    off_by_one = ADL.compute_cost(
        unkeyed,
        X,
        DataFrame(load_a = [18.0, 12.0], load_b = [11.0, 4.0]),
        false,
        false,
    )
    @test !isapprox(off_by_one, _LINK_COST; atol = 1e-6)
end

@testset "6. matrices, with the Y column order written down" begin
    # Variation 1 again, except that the `outputs` say which column of `Y` holds
    # which variable. Same problem, same cost - the difference is that the layout
    # is now declared instead of inferred from declaration order, so inserting a
    # `@variable` above these two cannot silently change what column 1 means.
    model = ADL.Model()
    @variable(model, g1 >= 0, ADL.Policy)
    @variable(model, g2 >= 0, ADL.Policy)
    @variable(model, load_a, ADL.Forecast)
    @variable(model, load_b, ADL.Forecast)

    @constraints(ADL.Plan(model), begin
        g1.plan >= load_a.plan
        g2.plan >= load_b.plan
    end)
    @objective(ADL.Plan(model), Min, 10 * g1.plan + 20 * g2.plan)

    @variables(ADL.Assess(model), begin
        s1 >= 0
        s2 >= 0
    end)
    @constraints(ADL.Assess(model), begin
        s1 >= load_a.assess - g1.assess
        s2 >= load_b.assess - g2.assess
    end)
    @objective(
        ADL.Assess(model),
        Min,
        10 * g1.assess + 20 * g2.assess + 50 * s1 + 30 * s2
    )

    set_optimizer(model, HiGHS.Optimizer)
    set_silent(model)
    # `Y` arrives with the loads the other way round, and this says so
    ADL.set_forecast_model(
        model,
        ADL.ForecastModel(
            architecture = Chain(
                Dense(2 => 2; bias = false, init = _identity_2x2),
            ),
            outputs = [load_a => 2, load_b => 1],
        ),
    )
    @test model.forecast.output_columns == [2, 1]

    X = [
        10.0 5.0
        20.0 8.0
    ]
    # column 1 is load_b, column 2 is load_a - the reverse of variation 1
    Y = [
        4.0 12.0
        11.0 18.0
    ]

    @test ADL.compute_cost(model, X, Y, false, false) ≈ _LINK_COST atol = 1e-6
    @test ADL.compute_cost(model, X, Y) ≈ sum(_LINK_COST) / 2 atol = 1e-6

    # and the declaration is what makes it right: variation 1's column order now
    # produces a different number instead of quietly being accepted
    @test !isapprox(
        ADL.compute_cost(model, X, Y[:, [2, 1]], false, false),
        _LINK_COST;
        atol = 1e-6,
    )

    # a gap is allowed, exactly as it is for `inputs`: the unread column has to be
    # there and is not looked at
    gapped = ADL.Model()
    @variable(gapped, h1 >= 0, ADL.Policy)
    @variable(gapped, h2 >= 0, ADL.Policy)
    @variable(gapped, load_a_g, ADL.Forecast)
    @variable(gapped, load_b_g, ADL.Forecast)
    @constraints(ADL.Plan(gapped), begin
        h1.plan >= load_a_g.plan
        h2.plan >= load_b_g.plan
    end)
    @objective(ADL.Plan(gapped), Min, 10 * h1.plan + 20 * h2.plan)
    @variables(ADL.Assess(gapped), begin
        t1 >= 0
        t2 >= 0
    end)
    @constraints(ADL.Assess(gapped), begin
        t1 >= load_a_g.assess - h1.assess
        t2 >= load_b_g.assess - h2.assess
    end)
    @objective(
        ADL.Assess(gapped),
        Min,
        10 * h1.assess + 20 * h2.assess + 50 * t1 + 30 * t2
    )
    set_optimizer(gapped, HiGHS.Optimizer)
    set_silent(gapped)
    ADL.set_forecast_model(
        gapped,
        ADL.ForecastModel(
            architecture = Chain(
                Dense(2 => 2; bias = false, init = _identity_2x2),
            ),
            outputs = [load_a_g => 1, load_b_g => 3],
        ),
    )
    Yg = [
        12.0 -999.0 4.0
        18.0 -999.0 11.0
    ]
    @test ADL.compute_cost(gapped, X, Yg, false, false) ≈ _LINK_COST atol = 1e-6
    # ... and a `Y` that does not reach column 3 is an error rather than a guess
    @test_throws ArgumentError ADL.compute_cost(
        gapped,
        X,
        Yg[:, 1:2],
        false,
        false,
    )

    # a named container cannot be matched against positions: its columns have
    # names, and matching those by order is what the rule refuses
    @test_throws ArgumentError ADL.compute_cost(
        model,
        X,
        DataFrame(load_a = [12.0, 18.0], load_b = [4.0, 11.0]),
        false,
        false,
    )
end
