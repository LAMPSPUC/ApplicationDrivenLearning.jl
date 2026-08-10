# Tests for the opt-in `@timeit_debug` instrumentation of the hot path.
#
# The instrumentation is internal, so everything here is reached through
# qualified `_`-prefixed names rather than exports.
#
# `TimerOutputs` is reached through the package rather than added as a test
# dependency, so that these tests always exercise the exact version the package
# resolves to.
const TO = ApplicationDrivenLearning.TimerOutputs

# Section names recorded at the top level of the package timer.
_timer_sections(to) = keys(TO.todict(to)["inner_timers"])

@testset "Timing instrumentation" begin
    tmodel = ApplicationDrivenLearning.Model()
    @variables(tmodel, begin
        x >= 0, ApplicationDrivenLearning.Policy
        d, ApplicationDrivenLearning.Forecast
    end)
    @objective(ApplicationDrivenLearning.Plan(tmodel), Min, x.plan)
    @objective(ApplicationDrivenLearning.Assess(tmodel), Min, x.assess)
    set_optimizer(tmodel, HiGHS.Optimizer)
    set_silent(tmodel)
    ApplicationDrivenLearning.set_forecast_model(tmodel, Chain(Dense(1 => 1)))

    Tt = 3
    Xt = Float32.(ones(Tt, 1))
    Yt = Float32.(ones(Tt, 1))

    to = ApplicationDrivenLearning._timer()
    @test to isa TO.TimerOutput

    # the sections must record nothing until they are explicitly enabled
    ApplicationDrivenLearning._reset_timer!()
    ApplicationDrivenLearning.compute_cost(tmodel, Xt, Yt)
    @test isempty(_timer_sections(to))

    ApplicationDrivenLearning._enable_timing!()
    try
        ApplicationDrivenLearning._reset_timer!()
        ApplicationDrivenLearning.compute_cost(tmodel, Xt, Yt)

        sections = _timer_sections(to)
        @test "sample_loop" in sections
        @test "forward_pass" in sections

        # one sweep over the samples, one plan and one assess solve per sample
        @test TO.ncalls(to["sample_loop"]) == 1
        @test TO.ncalls(to["sample_loop"]["plan_solve"]) == Tt
        @test TO.ncalls(to["sample_loop"]["assess_solve"]) == Tt

        # a second sweep accumulates instead of replacing
        ApplicationDrivenLearning.compute_cost(tmodel, Xt, Yt)
        @test TO.ncalls(to["sample_loop"]) == 2
        @test TO.ncalls(to["sample_loop"]["plan_solve"]) == 2Tt

        # the gradient sections only appear on the gradient path
        @test !(
            "diffopt_reverse" in
            keys(TO.todict(to["sample_loop"])["inner_timers"])
        )
        ApplicationDrivenLearning.compute_cost(tmodel, Xt, Yt, true)
        @test TO.ncalls(to["sample_loop"]["diffopt_reverse"]) == Tt

        buf = IOBuffer()
        ApplicationDrivenLearning._print_timer(buf)
        @test occursin("sample_loop", String(take!(buf)))

        ApplicationDrivenLearning._reset_timer!()
        @test isempty(_timer_sections(to))
    finally
        ApplicationDrivenLearning._disable_timing!()
    end

    # ... and stop recording again once disabled
    ApplicationDrivenLearning._reset_timer!()
    ApplicationDrivenLearning.compute_cost(tmodel, Xt, Yt)
    @test isempty(_timer_sections(to))
end
