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
