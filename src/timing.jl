# Opt-in timing instrumentation for the internal hot path.
#
# This file holds all of the package's coupling to TimerOutputs: the imports,
# the timer object and the small API around it. The `@timeit_debug` sections
# themselves live next to the code they measure, in `simulation.jl` and
# `forecast_model.jl`.
#
# It has to be included before those files: `@timeit_debug` and `_TIMER` must be
# resolvable when they are expanded.
#
# Everything here is internal. It exists to profile the package's own hot path,
# not to time user models, so none of it is exported or part of the public API.

import TimerOutputs
using TimerOutputs: @timeit_debug

"""
    ApplicationDrivenLearning._TIMER

`TimerOutputs.TimerOutput` collecting the timings of the internal hot path
(plan/assess solves, DiffOpt reverse pass, predictive model forward and
backward passes, parameter flattening).

The sections are declared with `@timeit_debug`, which compiles to nothing
unless [`_enable_timing!`](@ref) has been called, so instrumentation is free for
regular use.
"""
const _TIMER = TimerOutputs.TimerOutput()

"""
    _enable_timing!()

Turn on the internal `@timeit_debug` sections. Triggers recompilation of the
instrumented methods, so call it once before the workload to be measured.

```julia
ApplicationDrivenLearning._enable_timing!()
ApplicationDrivenLearning.train!(model, X, Y, options)  # warm up
ApplicationDrivenLearning._reset_timer!()
ApplicationDrivenLearning.train!(model, X, Y, options)  # measure
ApplicationDrivenLearning._print_timer()
```
"""
_enable_timing!() = TimerOutputs.enable_debug_timings(ApplicationDrivenLearning)

"""
    _disable_timing!()

Turn the internal `@timeit_debug` sections back off. See
[`_enable_timing!`](@ref).
"""
function _disable_timing!()
    return TimerOutputs.disable_debug_timings(ApplicationDrivenLearning)
end

"""
    _reset_timer!()

Clear every section recorded in [`_TIMER`](@ref).
"""
_reset_timer!() = TimerOutputs.reset_timer!(_TIMER)

"""
    _timer()

Return the [`_TIMER`](@ref) object, for programmatic inspection with the
`TimerOutputs` API.
"""
_timer() = _TIMER

"""
    _print_timer(io=stdout; kwargs...)

Print the [`_TIMER`](@ref) table. Keyword arguments are forwarded to
`TimerOutputs.print_timer`.
"""
function _print_timer(io::IO = stdout; kwargs...)
    return TimerOutputs.print_timer(io, _TIMER; kwargs...)
end
