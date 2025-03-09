abstract type AbstractOptimizationMode end

"""
    BilevelMode <: AbstractOptimizationMode

Used to solve the application driven learning training problem as a bilevel optimization problem
by using the BilevelJuMP.jl package.

...
# Parameters
- `optimizer::Function` is equivalent to `solver` in BilevelJuMP.BilevelModel.
- `silent::Bool` is equivalent to `silent` in BilevelJuMP.BilevelModel.
- `mode::Union{Nothing, BilevelJuMP.BilevelMode}` is equivalent to `mode` in BilevelJuMP.BilevelModel. 
...
"""
struct BilevelMode <: AbstractOptimizationMode end

"""
    NelderMeadMode <: AbstractOptimizationMode

Used to solve the application driven learning training problem using the Nelder-Mead optimization method
implementation from Optim.jl package.

...
# Parameters
- `initial_simplex` is the initial simplex of solutions to be applied.
- `parameters` is the parameters to be applied to the Nelder-Mead optimization method.
...
"""
struct NelderMeadMode <: AbstractOptimizationMode end

"""
    GradientMode <: AbstractOptimizationMode

Used to solve the application driven learning training problem using the gradient optimization method

...
# Parameters
- `rule` is the optimiser object to be used in the gradient optimization process.
- 'epochs' is the number of epochs to be used in the gradient optimization process.
- 'batch_size' is the batch size to be used in the gradient optimization process.
- 'verbose' is the flag of whether to print the training process.
- 'compute_cost_every' is the epoch frequency for computing the cost and evaluating best solution.
- 'time_limit' is the time limit for the training process.
...
"""
struct GradientMode <: AbstractOptimizationMode end

"""
    NelderMeadMPIMode <: AbstractOptimizationMode

MPI implementation of NelderMeadMode.
"""
struct NelderMeadMPIMode <: AbstractOptimizationMode end

"""
    GradientMPIMode <: AbstractOptimizationMode

MPI implementation of GradientMode.
"""
struct GradientMPIMode <: AbstractOptimizationMode end

"""
    Options(mode; params...)

Options struct to hold optimization mode and mode parameters.

...
# Example
```julia
options = Options(
    GradientMode; 
    rule=Optim.RMSProp(0.01),
    epochs=100,
    batch_size=10
)
```
...
"""
struct Options
    mode
    params::Dict{Symbol, Any}

    function Options(mode; params...)
        if mode <: AbstractOptimizationMode
            return new(mode, Dict(params))
        else
            throw(ArgumentError("mode must be a subtype of AbstractOptimizationMode"))
        end
    end
end
