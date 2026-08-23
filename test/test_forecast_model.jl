# Unit-level tests of the forecast itself: how `ForecastModel` units assemble
# into a `FullForecastModel`, what it predicts, and how its parameters are read,
# written and updated.
#
# The forecasts here are never attached to an `ApplicationDrivenLearning.Model` -
# `FullForecastModel` is built directly - so the coverage checks that live in
# `set_forecast_model` are not in play. `test_api.jl` covers those.
in_size = 3
out_size = 2

# forecast variables to wire the units to. A unit says what it predicts, so even a
# standalone forecast needs variables to name
_fcst_model = ApplicationDrivenLearning.Model()
@variables(_fcst_model, begin
    f[1:out_size], ApplicationDrivenLearning.Forecast
end)

@testset "Single-Dense" begin
    forecaster = ApplicationDrivenLearning.FullForecastModel([
        ApplicationDrivenLearning.ForecastModel(
            architecture = Flux.Dense(in_size => out_size) |> f64,
            outputs = [f[1], f[2]],
        ),
    ])
    # `inputs = nothing` on the only unit: the width comes from the architecture
    @test forecaster.input_size == in_size
    # the one unit reads every column and writes both rows, resolved once
    @test forecaster.units[1].inputs == [1, 2, 3]
    @test forecaster.units[1].outputs == [f[1], f[2]]
    @test forecaster.units[1].rows == [1, 2]

    x = ones((in_size, 1))
    @test size(forecaster(x)) == (out_size, 1)

    θ = ApplicationDrivenLearning.extract_params(forecaster)
    expected_params_size = in_size * out_size + out_size
    @test size(θ) == (expected_params_size,)

    ApplicationDrivenLearning.apply_params(
        forecaster,
        ones(expected_params_size),
    )
    x = ones(in_size)
    @test forecaster(x) == (in_size + 1) .* ones(out_size)

    ApplicationDrivenLearning.apply_gradient!(
        forecaster,
        ones((1, out_size)),
        ones((1, in_size)),
        Flux.setup(Flux.Descent(0.1), forecaster),
    )
    @test Flux.trainables(forecaster)[1] == 0.9 * ones((out_size, in_size))
    @test Flux.trainables(forecaster)[2] == 0.9 * ones(out_size)
end

@testset "Single-Chain" begin
    forecaster = ApplicationDrivenLearning.FullForecastModel([
        ApplicationDrivenLearning.ForecastModel(
            architecture = Flux.Chain(Flux.Dense(in_size => out_size) |> f64),
            outputs = [f[1], f[2]],
        ),
    ])
    x = ones((in_size, 1))
    @test size(forecaster(x)) == (out_size, 1)

    θ = ApplicationDrivenLearning.extract_params(forecaster)
    expected_params_size = in_size * out_size + out_size
    @test size(θ) == (expected_params_size,)

    ApplicationDrivenLearning.apply_params(
        forecaster,
        ones(expected_params_size),
    )
    x = ones(in_size)
    @test forecaster(x) == (in_size + 1) .* ones(out_size)

    ApplicationDrivenLearning.apply_gradient!(
        forecaster,
        ones((1, out_size)),
        ones((1, in_size)),
        Flux.setup(Flux.Descent(0.1), forecaster),
    )
    @test Flux.trainables(forecaster)[1] == 0.9 * ones((out_size, in_size))
    @test Flux.trainables(forecaster)[2] == 0.9 * ones(out_size)
end

@testset "One architecture reused across units" begin
    # The case the unit API exists for: the same architecture predicting two
    # variables from two different column pairs. Each unit gets its own copy, so
    # there are two independent parameter sets and the two train apart - which is
    # what "the same architecture" is taken to mean.
    model_in_size = 2
    model_out_size = 1

    nn = Flux.Dense(model_in_size => model_out_size) |> f64
    forecaster = ApplicationDrivenLearning.FullForecastModel([
        ApplicationDrivenLearning.ForecastModel(
            inputs = [1, 2],
            architecture = nn,
            outputs = [f[1]],
        ),
        ApplicationDrivenLearning.ForecastModel(
            inputs = [1, 3],
            architecture = deepcopy(nn),
            outputs = [f[2]],
        ),
    ])

    x = ones((in_size, 1))
    @test size(forecaster(x)) == (out_size, 1)

    # two parameter sets, not one: this is the number that says the units are
    # independent
    θ = ApplicationDrivenLearning.extract_params(forecaster)
    expected_params_size = 2 * (model_in_size * model_out_size + model_out_size)
    @test size(θ) == (expected_params_size,)

    ApplicationDrivenLearning.apply_params(
        forecaster,
        ones(expected_params_size),
    )
    x = ones(in_size)
    @test forecaster(x) == (model_in_size + 1) .* ones(out_size)

    # asymmetric cost gradients, so that a shared parameter set could not produce
    # this result: unit 1 sees dC/dŷ = 1 and unit 2 sees 2, and `Descent(0.1)`
    # moves them by 0.1 and 0.2 respectively
    ApplicationDrivenLearning.apply_gradient!(
        forecaster,
        [1.0 2.0],
        ones((1, in_size)),
        Flux.setup(Flux.Descent(0.1), forecaster),
    )
    @test Flux.trainables(forecaster)[1] ==
          0.9 * ones((model_out_size, model_in_size))
    @test Flux.trainables(forecaster)[2] == 0.9 * ones(model_out_size)
    @test Flux.trainables(forecaster)[3] ==
          0.8 * ones((model_out_size, model_in_size))
    @test Flux.trainables(forecaster)[4] == 0.8 * ones(model_out_size)

    # and handing the *same* object to both units is refused rather than quietly
    # tying their weights together
    err = try
        ApplicationDrivenLearning.FullForecastModel([
            ApplicationDrivenLearning.ForecastModel(
                inputs = [1, 2],
                architecture = nn,
                outputs = [f[1]],
            ),
            ApplicationDrivenLearning.ForecastModel(
                inputs = [1, 3],
                architecture = nn,
                outputs = [f[2]],
            ),
        ])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("same `architecture` object", err.msg)
    @test occursin("deepcopy", err.msg)
end

@testset "Multi-Model-Dense" begin
    model_in_size = 2
    model_out_size = 1

    nn1 = Flux.Dense(model_in_size => model_out_size) |> f64
    nn2 = Flux.Dense(model_in_size => model_out_size) |> f64
    forecaster = ApplicationDrivenLearning.FullForecastModel([
        ApplicationDrivenLearning.ForecastModel(
            inputs = [1, 2],
            architecture = nn1,
            outputs = [f[1]],
        ),
        ApplicationDrivenLearning.ForecastModel(
            inputs = [1, 3],
            architecture = nn2,
            outputs = [f[2]],
        ),
    ])

    x = ones((in_size, 1))
    @test size(forecaster(x)) == (out_size, 1)

    θ = ApplicationDrivenLearning.extract_params(forecaster)
    expected_params_size = 2 * (model_in_size * model_out_size + model_out_size)
    @test size(θ) == (expected_params_size,)

    ApplicationDrivenLearning.apply_params(
        forecaster,
        ones(expected_params_size),
    )
    x = ones(in_size)
    @test forecaster(x) == (model_in_size + 1) .* ones(out_size)

    ApplicationDrivenLearning.apply_gradient!(
        forecaster,
        ones((1, out_size)),
        ones((1, in_size)),
        Flux.setup(Flux.Descent(0.1), forecaster),
    )
    @test Flux.trainables(forecaster)[1] ==
          0.9 * ones((model_out_size, model_in_size))
    @test Flux.trainables(forecaster)[2] == 0.9 * ones(model_out_size)
    @test Flux.trainables(forecaster)[3] ==
          0.9 * ones((model_out_size, model_in_size))
    @test Flux.trainables(forecaster)[4] == 0.9 * ones(model_out_size)
end
