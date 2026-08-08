# Launches the MPI training modes under `mpiexec`.
#
# These modes cannot be exercised in-process: they need a real MPI
# communicator with a controller rank and at least one worker rank. The driver
# script does the assertions and encodes the outcome in its exit code.

using MPI

const MPI_DRIVER = joinpath(@__DIR__, "mpi", "mpi_modes.jl")
const MPI_NPROCS = 3

"""
Number of MPI ranks to launch, overridable with `ADL_MPI_NPROCS` and capped by
the number of available CPU threads so the run does not oversubscribe.
"""
function mpi_nprocs()
    requested = tryparse(Int, get(ENV, "ADL_MPI_NPROCS", ""))
    n = something(requested, MPI_NPROCS)
    # one controller + at least one worker
    return max(2, min(n, max(2, Sys.CPU_THREADS)))
end

"""
Propagate the parent's `--code-coverage` setting to the MPI ranks, so that the
distributed optimizers show up in the coverage report instead of appearing
dead. Each rank writes its own `<file>.<pid>.cov`, so the ranks do not clash.
"""
function coverage_flag()
    level = Base.JLOptions().code_coverage
    level == 1 && return `--code-coverage=user`
    level == 2 && return `--code-coverage=all`
    return ``
end

@testset "MPI training modes" begin
    nprocs = mpi_nprocs()
    project = dirname(Base.active_project())
    cmd = `$(MPI.mpiexec()) -n $nprocs $(Base.julia_cmd()) $(coverage_flag()) --project=$project $MPI_DRIVER`

    @info "Launching MPI modes test" nprocs driver = MPI_DRIVER
    proc = run(ignorestatus(cmd))
    @test proc.exitcode == 0
end
