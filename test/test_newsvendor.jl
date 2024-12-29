c = 5.0
q = 9.0
r = 4.0
X = ones(1, 1)
Y = 50 * ones(1, 1)
best_decision = y = Y[1, 1]
best_cost = (c-q)*y

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
@objective(ApplicationDrivenLearning.Plan(model), Min, c*x.plan-q*y-r*w)
@variables(ApplicationDrivenLearning.Assess(model), begin
    y >= 0
    w >= 0
end)
@constraints(ApplicationDrivenLearning.Assess(model), begin
    con1, y <= d.assess
    con2, y + w <= x.assess
end)
@objective(ApplicationDrivenLearning.Assess(model), Min, c*x.assess-q*y-r*w)
set_optimizer(model, HiGHS.Optimizer)
set_silent(model)
nn = Chain(
    Dense(1 => 1; bias=false, init=(size...) -> rand(size...)),
)

@testset "Newsvendor BilevelMode" begin
    ApplicationDrivenLearning.set_forecast_model(
        model,
        Chain(
            Dense(1 => 1; bias=false, init=(size...) -> rand(size...)),
        )
    )
    opt = ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.BilevelMode, 
        optimizer=HiGHS.Optimizer,
        mode=BilevelJuMP.FortunyAmatMcCarlMode(primal_big_M=100, dual_big_M=100)
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    @test sol.params[1] ≈ best_decision atol=1e-2
    @test sol.cost ≈ best_cost atol=1e-2
end

@testset "Newsvendor NelderMeadMode" begin
    ApplicationDrivenLearning.set_forecast_model(
        model,
        Chain(
            Dense(1 => 1; bias=false, init=(size...) -> rand(size...)),
        )
    )
    opt = ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.NelderMeadMode, 
        iterations=100,
        time_limit=60,
        show_trace=false
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    @test sol.params[1] ≈ best_decision atol=1e-2
    @test sol.cost ≈ best_cost atol=1e-2
end

@testset "Newsvendor GradientMode" begin
    ApplicationDrivenLearning.set_forecast_model(
        model,
        Chain(
            Dense(1 => 1; bias=false, init=(size...) -> rand(size...)),
        )
    )
    opt = ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.GradientMode;
        rule=Flux.Adam(1.0), 
        epochs=150,
        batch_size=-1, 
        verbose=false
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    @test sol.params[1] ≈ best_decision atol=1e-2
    @test sol.cost ≈ best_cost atol=1e-2
end