# TODO: create tests on predictive model functionalities
in_size = 3
out_size = 2

@testset "Single-Dense" begin
    forecaster = ApplicationDrivenLearning.PredictiveModel(
        Flux.Dense(in_size => out_size) |> f64,
    )
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
        ones(out_size),
        ones((1, in_size)),
        Flux.setup(Flux.Descent(0.1), forecaster),
    )
    @test Flux.params(forecaster.networks[1])[1] ==
          0.9 * ones((out_size, in_size))
    @test Flux.params(forecaster.networks[1])[2] == 0.9 * ones(out_size)
end

@testset "Single-Chain" begin
    forecaster = ApplicationDrivenLearning.PredictiveModel(
        Flux.Chain(Flux.Dense(in_size => out_size) |> f64),
    )
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
        ones(out_size),
        ones((1, in_size)),
        Flux.setup(Flux.Descent(0.1), forecaster),
    )
    @test Flux.params(forecaster.networks[1])[1] ==
          0.9 * ones((out_size, in_size))
    @test Flux.params(forecaster.networks[1])[2] == 0.9 * ones(out_size)
end

@testset "Multi-Variate-Dense" begin
    model_in_size = 2
    model_out_size = 1
    nn = Flux.Dense(model_in_size => model_out_size) |> f64
    in_out_map = Dict([1, 2] => [1], [1, 3] => [2])
    forecaster = ApplicationDrivenLearning.PredictiveModel(nn, in_out_map)
    x = ones((in_size, 1))
    @test size(forecaster(x)) == (out_size, 1)

    θ = ApplicationDrivenLearning.extract_params(forecaster)
    expected_params_size = model_in_size * model_out_size + model_out_size
    @test size(θ) == (expected_params_size,)

    ApplicationDrivenLearning.apply_params(
        forecaster,
        ones(expected_params_size),
    )
    x = ones(in_size)
    @test forecaster(x) == (model_in_size + 1) .* ones(out_size)

    ApplicationDrivenLearning.apply_gradient!(
        forecaster,
        ones(out_size),
        ones((1, in_size)),
        Flux.setup(Flux.Descent(0.1), forecaster),
    )
    @test Flux.params(forecaster.networks[1])[1] ==
          0.8 * ones((model_out_size, model_in_size))
    @test Flux.params(forecaster.networks[1])[2] == 0.8 * ones(model_out_size)
end

@testset "Multi-Model-Dense" begin
    model_in_size = 2
    model_out_size = 1
    nn1 = Flux.Dense(model_in_size => model_out_size) |> f64
    nn2 = Flux.Dense(model_in_size => model_out_size) |> f64
    in_out_map = [Dict([1, 2] => [1]), Dict([1, 3] => [2])]
    forecaster = ApplicationDrivenLearning.PredictiveModel(
        [nn1, nn2],
        in_out_map,
        in_size,
        out_size,
    )

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
        ones(out_size),
        ones((1, in_size)),
        Flux.setup(Flux.Descent(0.1), forecaster),
    )
    @test Flux.params(forecaster.networks[1])[1] ==
          0.9 * ones((model_out_size, model_in_size))
    @test Flux.params(forecaster.networks[1])[2] == 0.9 * ones(model_out_size)
    @test Flux.params(forecaster.networks[2])[1] ==
          0.9 * ones((model_out_size, model_in_size))
    @test Flux.params(forecaster.networks[2])[2] == 0.9 * ones(model_out_size)
end
