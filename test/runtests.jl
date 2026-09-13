using ApplicationDrivenLearning
using Flux
using JuMP
using HiGHS
using Optim
using BilevelJuMP
using Test
using Random

Random.seed!(123)

include("test_api.jl")
include("test_data_inputs.jl")
include("test_data_linking.jl")
include("test_predictive_model.jl")
include("test_newsvendor.jl")
include("test_gradient.jl")
include("test_custom_variables.jl")
include("test_timing.jl")

# The MPI modes spawn a separate `mpiexec` job, which is slow and needs a
# working MPI runtime. Set ADL_SKIP_MPI_TESTS=true to skip them.
if lowercase(get(ENV, "ADL_SKIP_MPI_TESTS", "false")) in ("1", "true")
    @info "Skipping MPI tests (ADL_SKIP_MPI_TESTS is set)"
else
    include("test_mpi.jl")
end
