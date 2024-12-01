"""
The umbrella problem

z: 1 if take the umbrella, 0 otherwise
y: 1 if it rains, 0 otherwise
C = 0.5 * z + 2 * y * (1-z)

C = 0.5 * z + k
k >= 2 - bigM * (1-y) - bigM * z

y=0, z=0 -> C=0.0
y=1, z=0 -> C=2.0
y=0, z=1 -> C=0.5
y=1, z=1 -> C=0.5

E[C | Prob(y=1)=p & z=0] = 2p
E[C | Prob(y=1)=p & z=1] = 0.5
"""
bigM = 100
c1 = 0.5  # cost of taking the umbrella
c2 = 2.0  # cost of not taking the umbrella when it rains
T = 30
f = 1
X = ones((T, f))
Y = rand(0:1, (T, 1))
best_decision = 1.0
best_cost = 0.5

model = ApplicationDrivenLearning.Model()
@variable(model, z, ApplicationDrivenLearning.Policy, Bin)
@variable(model, y, ApplicationDrivenLearning.Forecast)

@testset "Binary variables" begin
    @test JuMP.is_binary(z.plan)
    @test JuMP.is_binary(z.assess)
end

@variables(ApplicationDrivenLearning.Plan(model), begin
    k >= 0
    w, Bin
end)
@constraints(ApplicationDrivenLearning.Plan(model), begin
    w >= c2 * (y.plan - 0.5)
    w <= y.plan + 0.5
    k >= c2 - bigM*(1-w) - bigM*z.plan
end)
@objective(ApplicationDrivenLearning.Plan(model), Min, c1*z.plan + k)
@variables(ApplicationDrivenLearning.Assess(model), begin
    k >= 0
    w, Bin
end)
@constraints(ApplicationDrivenLearning.Assess(model), begin
    w >= 2*(y.assess - 0.5)
    w <= y.assess + 0.5
    k >= c2 - bigM*(1-w) - bigM*z.assess
end)
@objective(ApplicationDrivenLearning.Assess(model), Min, c1*z.assess + k)
set_optimizer(model, HiGHS.Optimizer, false)
set_silent(model)

@testset "Umbrella NelderMeadModel No Rain" begin
    Y = zeros((T, 1))
    nn = Chain(
        Dense(f => 1, Flux.sigmoid; bias=false, init=(size...) -> rand(size...) .- 0.4)
    )
    ApplicationDrivenLearning.set_forecast_model(model, nn)
    @test mean(model.forecast(X')) > 0.5
    opt = ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.NelderMeadMode,
        initial_simplex=Optim.AffineSimplexer(a=0.0, b=100.0),
        iterations=100,
        show_trace=true,
        time_limit=60,
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    @test mean(model.forecast(X')) < 0.5
    @test sol.cost / T ≈ 0.0 atol=1e-2
end

@testset "Umbrella NelderMeadModel Always Rain" begin
    Y = ones((T, 1))
    nn = Chain(
        Dense(f => 1, Flux.sigmoid; bias=false, init=(size...) -> rand(size...) .- 0.6)
    )
    ApplicationDrivenLearning.set_forecast_model(model, nn)
    @test mean(model.forecast(X')) < 0.5
    opt = ApplicationDrivenLearning.Options(
        ApplicationDrivenLearning.NelderMeadMode,
        initial_simplex=Optim.AffineSimplexer(a=0.0, b=100.0),
        iterations=100,
        show_trace=true,
        time_limit=60,
    )
    sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
    @test mean(model.forecast(X')) > 0.5
    @test sol.cost / T ≈ 0.5 atol=1e-2
end
