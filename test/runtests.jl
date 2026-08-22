using ApplicationDrivenLearning
using Flux
using JuMP
using HiGHS
using Optim
using BilevelJuMP
using Test
using Random

Random.seed!(123)

"""
Propagate the parent's `--code-coverage` setting to any process a test spawns, so
that code which only ever runs off this process — the MPI ranks and the
`Distributed` workers — shows up in the coverage report instead of appearing dead.
Each spawned process writes its own `<file>.<pid>.cov`, so they do not clash.

Returns an empty `Cmd` when coverage is off, which interpolates into a command
without adding an argument.
"""
function coverage_flag()
    level = Base.JLOptions().code_coverage
    level == 1 && return `--code-coverage=user`
    level == 2 && return `--code-coverage=all`
    return ``
end

include("test_api.jl")
include("test_data_inputs.jl")
include("test_data_linking.jl")
include("test_solver_backends.jl")
include("test_predictive_model.jl")
include("test_newsvendor.jl")
include("test_gradient.jl")
include("test_custom_variables.jl")
include("test_timing.jl")

# Last of the in-process tests: it adds worker processes and removes them again, so
# it runs after the tests that measure timing and allocation on this process.
include("test_distributed.jl")

# The MPI modes spawn a separate `mpiexec` job, which is slow and needs a
# working MPI runtime. Set ADL_SKIP_MPI_TESTS=true to skip them.
if lowercase(get(ENV, "ADL_SKIP_MPI_TESTS", "false")) in ("1", "true")
    @info "Skipping MPI tests (ADL_SKIP_MPI_TESTS is set)"
else
    include("test_mpi.jl")
end
