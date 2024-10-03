abstract type AbstractOptimizationMode end

struct BilevelMode <: AbstractOptimizationMode end
struct NelderMeadMode <: AbstractOptimizationMode end
struct GradientMode <: AbstractOptimizationMode end
struct NelderMeadMPIMode <: AbstractOptimizationMode end
struct GradientMPIMode <: AbstractOptimizationMode end

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
