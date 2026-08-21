# `DistributedBackend`: the same per-sample evaluation the MPI backend distributes,
# spread over `Distributed` worker processes instead.
#
# Unlike the MPI tests this needs no external launcher and no MPI runtime, so it runs
# as part of the ordinary suite rather than behind ADL_SKIP_MPI_TESTS. That is what it
# buys: the parallel code paths - including the seeded stochastic-batch equality that
# test/mpi/mpi_modes.jl checks - are covered by a testset that cannot be switched off.
#
# It is not, however, free: ~145s, of which the checks are ~20s. The rest is fixed
# cost - `addprocs` (17s), loading Flux and HiGHS on two workers (24s), and each worker
# JIT-compiling the JuMP/DiffOpt/HiGHS solve path for itself on the first distributed
# solve (104s). Being fixed rather than per-check is why every check that fits lives
# here. For scale: the MPI testset it complements measures 6m22s.

using Distributed
using NLopt
using DataFrames

# `--project` or the workers come up without the test environment. `coverage_flag`
# (runtests.jl) is what keeps the worker-side code out of the dead-code column of the
# coverage report: it only ever runs off this process.
addprocs(2; exeflags = `--project=$(Base.active_project()) $(coverage_flag())`)

# Flux and HiGHS are needed on the workers because the *builder* uses them. NLopt and
# Optim are not: the optimizer only ever runs on the driver, which is a fair
# illustration of how little this backend actually distributes.
@everywhere using ApplicationDrivenLearning, Flux, JuMP, HiGHS

# newsvendor with c < q and r < c, so the optimal order quantity is the realized
# demand and the optimal per-sample cost is (c - q) * d
@everywhere const DIST_C = 5.0
@everywhere const DIST_Q = 9.0
@everywhere const DIST_R = 4.0

# Fully qualified rather than through the `ADL` alias the other test files use: this
# function runs on the workers, where only what `@everywhere` put there exists.
@everywhere function dist_newsvendor(
    w0::Float64 = 1.0;
    q = DIST_Q,
    n_in::Int = 1,
)
    m = ApplicationDrivenLearning.Model()
    @variables(m, begin
        x, ApplicationDrivenLearning.Policy
        d, ApplicationDrivenLearning.Forecast
    end)
    @variables(ApplicationDrivenLearning.Plan(m), begin
        yp >= 0
        wp >= 0
    end)
    @constraints(ApplicationDrivenLearning.Plan(m), begin
        yp <= d.plan
        yp + wp <= x.plan
    end)
    @objective(
        ApplicationDrivenLearning.Plan(m),
        Min,
        DIST_C * x.plan - q * yp - DIST_R * wp
    )
    @variables(ApplicationDrivenLearning.Assess(m), begin
        ya >= 0
        wa >= 0
    end)
    @constraints(ApplicationDrivenLearning.Assess(m), begin
        ya <= d.assess
        ya + wa <= x.assess
    end)
    @objective(
        ApplicationDrivenLearning.Assess(m),
        Min,
        DIST_C * x.assess - q * ya - DIST_R * wa
    )
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    ApplicationDrivenLearning.set_forecast_model(
        m,
        ApplicationDrivenLearning.ForecastModel(
            architecture = Chain(
                Dense(n_in => 1; bias = false, init = (s...) -> fill(w0, s)),
            ),
            outputs = [d => :demand],
        ),
    )
    return m, d
end

@everywhere dist_builder() = first(dist_newsvendor())

# Builders whose result depends on *where* they run. With the model and the workers
# both coming from one function, that is the only way the two can now disagree - and
# it is what `verify` guards against.
@everywhere dist_builder_worker_shape() =
    first(dist_newsvendor(; n_in = myid() == 1 ? 1 : 2))
@everywhere dist_builder_worker_cost() =
    first(dist_newsvendor(; q = myid() == 1 ? DIST_Q : 20.0))

const DIST_T = 4
const DIST_X = ones(DIST_T, 1)
const DIST_Y = [50.0, 40.0, 60.0, 50.0]

# named rather than positional: the column is matched to the `demand` forecast
# variable by the `output_names` the builder declares
dist_data() = (DIST_X, DataFrame(demand = DIST_Y))
dist_backend(; kwargs...) = ADL.DistributedBackend(; kwargs...)

# The first distributed evaluation makes each worker compile the whole
# JuMP/DiffOpt/HiGHS solve path for itself, which costs far more than any check
# below. Pay it once here so the testset timings report work rather than
# compilation - and note that this, not the launcher, is what makes an
# out-of-process parallel test expensive.
ADL.train!(
    dist_builder,
    dist_data()...,
    ADL.Options(
        ADL.GradientMode;
        epochs = 1,
        verbose = false,
        parallel = dist_backend(),
    ),
)

@testset "DistributedBackend" begin
    @testset "index splitting" begin
        # order and repeats are what keep the returned gradient rows aligned with
        # the batch, so neither may be normalized away
        @test ADL._split_indices([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
        @test ADL._split_indices([1, 2, 3, 4, 5], 2) == [[1, 2, 3], [4, 5]]
        @test ADL._split_indices([3, 1, 3], 2) == [[3, 1], [3]]
        # a batch smaller than the pool uses fewer workers rather than dispatching
        # empty chunks
        @test ADL._split_indices([7], 4) == [[7]]
        @test ADL._split_indices(1:3, 3) == [[1], [2], [3]]
        # every element appears exactly once, in order, however it is cut
        for n = 1:6
            @test reduce(vcat, ADL._split_indices(collect(1:5), n)) == 1:5
        end
    end

    @testset "a Model is refused, a builder is required" begin
        # The workers never run the caller's script and cannot be sent a model, so
        # this backend is the one case where `train!` needs the function form.
        m, _ = dist_newsvendor()
        err = try
            ADL.train!(
                m,
                dist_data()...,
                ADL.Options(
                    ADL.GradientMode;
                    epochs = 1,
                    verbose = false,
                    parallel = dist_backend(),
                ),
            )
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("needs a model builder", err.msg)
        @test occursin("train!(build_model", err.msg)

        # every other backend is satisfied by a model
        @test ADL.train!(
            m,
            dist_data()...,
            ADL.Options(ADL.GradientMode; epochs = 1, verbose = false),
        ) isa ADL.Solution
    end

    @testset "GradientMode matches serial" begin
        opts(extra...) = ADL.Options(
            ADL.GradientMode;
            rule = Flux.Adam(5.0),
            epochs = 20,
            verbose = false,
            extra...,
        )
        sol_serial = ADL.train!(dist_builder, dist_data()..., opts())
        sol_dist = ADL.train!(
            dist_builder,
            dist_data()...,
            opts(:parallel => dist_backend()),
        )

        @test sol_dist.cost ≈ sol_serial.cost atol = 1e-4
        @test sol_dist.params ≈ sol_serial.params atol = 1e-3
        @test sol_dist.cost < -4.0        # training improved on the start point

        # the documented way to recover a fitted model from the builder form: the
        # parameters are in the `Solution`, and they reproduce the reported cost
        fitted, _ = dist_newsvendor()
        ADL.apply_params(fitted.forecast, sol_dist.params)
        @test ADL.compute_cost(fitted, dist_data()...) ≈ sol_dist.cost atol =
            1e-6
    end

    @testset "GradientMode stochastic batches match serial" begin
        # Batches are drawn on the driver, from one `rand` call made before any work
        # is dispatched, so seeding identically makes the serial and distributed runs
        # see the same samples in the same order. This is the check that pins the
        # per-sample gradient rows to their batch rows.
        opts(extra...) = ADL.Options(
            ADL.GradientMode;
            rule = Flux.Adam(5.0),
            epochs = 20,
            batch_size = 2,
            verbose = false,
            extra...,
        )
        Random.seed!(42)
        sol_serial = ADL.train!(dist_builder, dist_data()..., opts())
        Random.seed!(42)
        sol_dist = ADL.train!(
            dist_builder,
            dist_data()...,
            opts(:parallel => dist_backend()),
        )

        @test sol_dist.cost ≈ sol_serial.cost atol = 1e-4
        @test sol_dist.params ≈ sol_serial.params atol = 1e-3
    end

    @testset "OptimMode matches serial" begin
        opts(extra...) = ADL.Options(
            ADL.OptimMode;
            algorithm = Optim.NelderMead(),
            iterations = 200,
            extra...,
        )
        sol_serial = ADL.train!(dist_builder, dist_data()..., opts())
        sol_dist = ADL.train!(
            dist_builder,
            dist_data()...,
            opts(:parallel => dist_backend()),
        )

        @test sol_dist.cost ≈ sol_serial.cost atol = 1e-4
        @test sol_dist.params ≈ sol_serial.params atol = 1e-3

        # Optimizing the right thing, not merely the same thing as serial. The
        # assessed cost is piecewise linear in the order quantity with breakpoints
        # at the observed demands, so scanning them is the exact optimum. Scanned
        # rather than hardcoded because it is not the obvious value: `DIST_R` makes
        # leftover stock cheap, so the optimum sits above the median demand.
        reference, _ = dist_newsvendor()
        best = minimum(DIST_Y) do candidate
            ADL.apply_params(reference.forecast, [candidate])
            return ADL.compute_cost(reference, dist_data()...)
        end
        @test sol_dist.cost ≈ best atol = 1e-4
    end

    @testset "NLoptMode matches serial (gradient-based)" begin
        # The case that needs the driver's own model kept at θ: the per-sample dC/dŷ
        # comes back from the workers, but turning it into dC/dθ happens on the
        # driver and reads the driver's parameters.
        opts(extra...) = ADL.Options(
            ADL.NLoptMode;
            algorithm = :LD_LBFGS,
            lower_bounds = [0.0],
            upper_bounds = [100.0],
            xtol_rel = 1e-10,
            maxeval = 200,
            extra...,
        )
        sol_serial = ADL.train!(dist_builder, dist_data()..., opts())
        sol_dist = ADL.train!(
            dist_builder,
            dist_data()...,
            opts(:parallel => dist_backend()),
        )

        @test sol_dist.cost ≈ sol_serial.cost atol = 1e-4
        @test sol_dist.params ≈ sol_serial.params atol = 1e-3
    end

    @testset "a builder that is not deterministic is caught" begin
        # One builder now produces both the driver's model and the workers', so they
        # can only disagree if the builder itself depends on where it runs. That is
        # a narrow failure mode, but a silent one: nothing else would notice.
        opts(; verify = true) = ADL.Options(
            ADL.GradientMode;
            epochs = 2,
            verbose = false,
            parallel = dist_backend(; verify = verify),
        )

        # a different predictive-model size on the workers: caught by the free
        # shape comparison, whether or not `verify` is on
        err = try
            ADL.train!(dist_builder_worker_shape, dist_data()..., opts())
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("input_size", err.msg)

        # same shapes, different cost coefficient: only the value check sees this
        err = try
            ADL.train!(dist_builder_worker_cost, dist_data()..., opts())
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("different cost", err.msg)

        # and with the value check off it goes through, which is what makes the
        # check worth having on by default
        sol = ADL.train!(
            dist_builder_worker_cost,
            dist_data()...,
            opts(; verify = false),
        )
        @test sol isa ADL.Solution
    end

    @testset "an empty worker pool warns but still answers" begin
        sol_serial = ADL.train!(
            dist_builder,
            dist_data()...,
            ADL.Options(ADL.GradientMode; epochs = 5, verbose = false),
        )

        sol_dist = @test_logs (:warn,) match_mode = :any ADL.train!(
            dist_builder,
            dist_data()...,
            ADL.Options(
                ADL.GradientMode;
                epochs = 5,
                verbose = false,
                parallel = dist_backend(; workers = [1]),
            ),
        )
        @test sol_dist.cost ≈ sol_serial.cost atol = 1e-4

        # an explicitly empty pool is a different thing from a missing one: there is
        # nowhere at all to evaluate, so it is an error rather than a warning
        @test_throws ArgumentError ADL.train!(
            dist_builder,
            dist_data()...,
            ADL.Options(
                ADL.GradientMode;
                epochs = 1,
                verbose = false,
                parallel = dist_backend(; workers = Int[]),
            ),
        )
    end

    @testset "workers are released when training ends" begin
        ADL.train!(
            dist_builder,
            dist_data()...,
            ADL.Options(
                ADL.GradientMode;
                epochs = 2,
                verbose = false,
                parallel = dist_backend(),
            ),
        )
        # no worker keeps a model between `train!` calls, so a later run with a
        # different model cannot inherit a stale one
        for w in workers()
            @test remotecall_fetch(
                () -> isnothing(ApplicationDrivenLearning._WORKER_STATE[]),
                w,
            )
        end

        # and a released worker says so rather than silently returning nothing,
        # which is what a restart mid-training would otherwise look like
        err = try
            remotecall_fetch(
                ApplicationDrivenLearning._worker_chunk,
                first(workers()),
                [1.0],
                [1],
                false,
            )
            nothing
        catch e
            e
        end
        @test !isnothing(err)
        @test occursin("holds no model", sprint(showerror, err))
    end

    @testset "the value check is skipped loudly when the driver cannot solve" begin
        # The driver never solves under this backend, so a model with no optimizer
        # attached is legitimate - but then the cost comparison cannot run. It has
        # to say so: silently dropping to a shapes-only check would leave a
        # nondeterministic builder undetected with no trace.
        @everywhere function dist_no_optimizer()
            # built the same way as `dist_newsvendor`, minus `set_optimizer`
            m2 = ApplicationDrivenLearning.Model()
            @variables(m2, begin
                x, ApplicationDrivenLearning.Policy
                d, ApplicationDrivenLearning.Forecast
            end)
            @variables(ApplicationDrivenLearning.Plan(m2), begin
                yp >= 0
                wp >= 0
            end)
            @constraints(
                ApplicationDrivenLearning.Plan(m2),
                begin
                    yp <= d.plan
                    yp + wp <= x.plan
                end
            )
            @objective(
                ApplicationDrivenLearning.Plan(m2),
                Min,
                DIST_C * x.plan - DIST_Q * yp - DIST_R * wp
            )
            @variables(ApplicationDrivenLearning.Assess(m2), begin
                ya >= 0
                wa >= 0
            end)
            @constraints(
                ApplicationDrivenLearning.Assess(m2),
                begin
                    ya <= d.assess
                    ya + wa <= x.assess
                end
            )
            @objective(
                ApplicationDrivenLearning.Assess(m2),
                Min,
                DIST_C * x.assess - DIST_Q * ya - DIST_R * wa
            )
            # the workers need a solver, the driver does not
            if myid() != 1
                set_optimizer(m2, HiGHS.Optimizer)
                set_silent(m2)
            end
            ApplicationDrivenLearning.set_forecast_model(
                m2,
                ApplicationDrivenLearning.ForecastModel(
                    architecture = Chain(
                        Dense(
                            1 => 1;
                            bias = false,
                            init = (s...) -> ones(s...),
                        ),
                    ),
                    outputs = [d => :demand],
                ),
            )
            return m2
        end

        sol = @test_logs (:warn,) match_mode = :any ADL.train!(
            dist_no_optimizer,
            dist_data()...,
            ADL.Options(
                ADL.GradientMode;
                epochs = 3,
                verbose = false,
                parallel = dist_backend(),
            ),
        )
        # training still works: only the driver's *check* needed a solver, the
        # evaluations themselves happen on the workers
        @test isfinite(sol.cost)
    end
end

rmprocs(workers())
