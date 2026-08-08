# Driver executed under `mpiexec` by test/test_mpi.jl.
#
# Every rank builds the same models and calls `train!`; inside `train!` the
# controller runs the optimization algorithm and the workers serve per-sample
# cost/gradient evaluations. Each MPI mode is compared against its serial
# counterpart run on the very same problem and starting point: the distributed
# implementation must reproduce the serial answer exactly (up to solver noise).
#
# Exits with code 0 when all checks pass and 1 otherwise, so the parent test
# can simply assert on the exit code.

using ApplicationDrivenLearning
using Flux
using JuMP
using HiGHS
using Random
import JobQueueMPI as JQM

const ADL = ApplicationDrivenLearning

# `train!` initializes MPI itself, but this script queries the rank before the
# first `train!` call, so it has to initialize first. `mpi_init` is idempotent.
JQM.mpi_init()

# Every `train!` below passes `mpi_finalize = false`: once MPI is finalized,
# `JQM.is_controller_process()` can no longer be called, and the checks need
# it. MPI is finalized explicitly at the very end instead.
const IS_CONTROLLER = JQM.is_controller_process()

# newsvendor with c < q and r < c, so the optimal order quantity equals the
# realized demand and the optimal per-sample cost is (c - q) * d
const C_COST = 5.0
const Q_COST = 9.0
const R_COST = 4.0

function newsvendor_model(w0::Float64)
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
    @objective(ADL.Plan(m), Min, C_COST * x.plan - Q_COST * yp - R_COST * wp)
    @variables(ADL.Assess(m), begin
        ya >= 0
        wa >= 0
    end)
    @constraints(ADL.Assess(m), begin
        ya <= d.assess
        ya + wa <= x.assess
    end)
    @objective(
        ADL.Assess(m),
        Min,
        C_COST * x.assess - Q_COST * ya - R_COST * wa
    )
    set_optimizer(m, HiGHS.Optimizer)
    set_silent(m)
    ADL.set_forecast_model(
        m,
        Chain(Dense(1 => 1; bias = false, init = (s...) -> fill(w0, s))),
    )
    return m, d
end

const T = 4
const X = ones(T, 1)
const Y_TRUE = [50.0, 40.0, 60.0, 50.0]

train_data(d) = (X, Dict(d => Y_TRUE))

const FAILURES = String[]

is_controller() = IS_CONTROLLER

"""
Record the outcome of a check. Only meaningful on the controller: workers get
a placeholder `Solution` back from `train!`, so `cond` and `detail` are thunks
and are never evaluated on a worker rank.
"""
function check(name::String, cond, detail = () -> "")
    is_controller() || return true
    ok = cond()
    if ok
        println("  PASS  $name")
    else
        println("  FAIL  $name  ", detail())
        push!(FAILURES, name)
    end
    return ok
end

# ---------------------------------------------------------------------------
# NelderMeadMPIMode vs NelderMeadMode
# ---------------------------------------------------------------------------
if is_controller()
    println("== NelderMeadMPIMode ==")
end

m_serial, d_serial = newsvendor_model(1.0)
sol_nm_serial = ADL.train!(
    m_serial,
    train_data(d_serial)...,
    ADL.Options(ADL.NelderMeadMode; iterations = 200, show_trace = false),
)

m_mpi, d_mpi = newsvendor_model(1.0)
sol_nm_mpi = ADL.train!(
    m_mpi,
    train_data(d_mpi)...,
    ADL.Options(ADL.NelderMeadMPIMode; iterations = 200, mpi_finalize = false),
)

check(
    "nelder-mead: MPI cost matches serial",
    () -> isapprox(sol_nm_mpi.cost, sol_nm_serial.cost; atol = 1e-4),
    () -> "mpi=$(sol_nm_mpi.cost) serial=$(sol_nm_serial.cost)",
)
check(
    "nelder-mead: MPI params match serial",
    () -> isapprox(sol_nm_mpi.params, sol_nm_serial.params; atol = 1e-3),
    () -> "mpi=$(sol_nm_mpi.params) serial=$(sol_nm_serial.params)",
)
"""
Best achievable mean assess cost, found by scanning the candidate order
quantities. For a newsvendor the optimum always sits on one of the observed
demands, so this is the exact optimum of the training problem.
"""
function best_reference_cost(model)
    Ymat = reshape(Y_TRUE, T, 1)
    return minimum(Y_TRUE) do candidate
        ADL.apply_params(model.forecast, [candidate])
        return ADL.compute_cost(model, X, Ymat)
    end
end

check(
    "nelder-mead: reached the optimal order quantity",
    () -> isapprox(sol_nm_mpi.cost, best_reference_cost(m_serial); atol = 1e-4),
    () -> "cost=$(sol_nm_mpi.cost) best=$(best_reference_cost(m_serial))",
)

# ---------------------------------------------------------------------------
# GradientMPIMode vs GradientMode (deterministic, full batch)
# ---------------------------------------------------------------------------
if is_controller()
    println("== GradientMPIMode (full batch) ==")
end

m_serial, d_serial = newsvendor_model(1.0)
sol_gd_serial = ADL.train!(
    m_serial,
    train_data(d_serial)...,
    ADL.Options(
        ADL.GradientMode;
        rule = Flux.Adam(5.0),
        epochs = 20,
        verbose = false,
    ),
)

m_mpi, d_mpi = newsvendor_model(1.0)
sol_gd_mpi = ADL.train!(
    m_mpi,
    train_data(d_mpi)...,
    ADL.Options(
        ADL.GradientMPIMode;
        rule = Flux.Adam(5.0),
        epochs = 20,
        verbose = false,
        mpi_finalize = false,
    ),
)

check(
    "gradient: MPI cost matches serial",
    () -> isapprox(sol_gd_mpi.cost, sol_gd_serial.cost; atol = 1e-4),
    () -> "mpi=$(sol_gd_mpi.cost) serial=$(sol_gd_serial.cost)",
)
check(
    "gradient: MPI params match serial",
    () -> isapprox(sol_gd_mpi.params, sol_gd_serial.params; atol = 1e-3),
    () -> "mpi=$(sol_gd_mpi.params) serial=$(sol_gd_serial.params)",
)
check(
    "gradient: training improved on the starting point",
    () -> sol_gd_mpi.cost < -4.0,
    () -> "cost=$(sol_gd_mpi.cost)",
)

# ---------------------------------------------------------------------------
# GradientMPIMode vs GradientMode (stochastic batches)
#
# Both paths draw their batches from the global RNG, so seeding identically
# before each call makes the two runs see the same samples in the same order.
# This exercises the per-sample gradient stacking, which must stay aligned
# with the batch rows.
# ---------------------------------------------------------------------------
if is_controller()
    println("== GradientMPIMode (stochastic) ==")
end

m_serial, d_serial = newsvendor_model(1.0)
Random.seed!(42)
sol_sgd_serial = ADL.train!(
    m_serial,
    train_data(d_serial)...,
    ADL.Options(
        ADL.GradientMode;
        rule = Flux.Adam(5.0),
        epochs = 20,
        batch_size = 2,
        verbose = false,
    ),
)

m_mpi, d_mpi = newsvendor_model(1.0)
Random.seed!(42)
sol_sgd_mpi = ADL.train!(
    m_mpi,
    train_data(d_mpi)...,
    ADL.Options(
        ADL.GradientMPIMode;
        rule = Flux.Adam(5.0),
        epochs = 20,
        batch_size = 2,
        verbose = false,
        mpi_finalize = false,
    ),
)

check(
    "stochastic gradient: MPI cost matches serial",
    () -> isapprox(sol_sgd_mpi.cost, sol_sgd_serial.cost; atol = 1e-4),
    () -> "mpi=$(sol_sgd_mpi.cost) serial=$(sol_sgd_serial.cost)",
)
check(
    "stochastic gradient: MPI params match serial",
    () -> isapprox(sol_sgd_mpi.params, sol_sgd_serial.params; atol = 1e-3),
    () -> "mpi=$(sol_sgd_mpi.params) serial=$(sol_sgd_serial.params)",
)

JQM.mpi_barrier()
JQM.mpi_finalize()

if is_controller()
    if isempty(FAILURES)
        println("ALL MPI CHECKS PASSED")
    else
        println("MPI CHECKS FAILED: ", join(FAILURES, ", "))
        exit(1)
    end
end
exit(0)
