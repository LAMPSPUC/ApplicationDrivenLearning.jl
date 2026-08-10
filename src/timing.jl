# Opt-in timing instrumentation for the internal hot path.
#
# This file holds all of the package's coupling to TimerOutputs: the imports,
# the timer object and the small API around it. The `@timeit_debug` sections
# themselves live next to the code they measure, in `simulation.jl` and
# `predictive_model.jl`.
#
# It has to be included before those files: `@timeit_debug` and `TIMER` must be
# resolvable when they are expanded.

import TimerOutputs
using TimerOutputs: @timeit_debug

"""
    ApplicationDrivenLearning.TIMER

`TimerOutputs.TimerOutput` collecting the timings of the internal hot path
(plan/assess solves, DiffOpt reverse pass, predictive model forward and
backward passes, parameter flattening).

The sections are declared with `@timeit_debug`, which compiles to nothing
unless [`enable_timing!`](@ref) has been called, so instrumentation is free for
regular use.
"""
const TIMER = TimerOutputs.TimerOutput()

"""
    enable_timing!()

Turn on the internal `@timeit_debug` sections. Triggers recompilation of the
instrumented methods, so call it once before the workload to be measured.

```julia
ApplicationDrivenLearning.enable_timing!()
ApplicationDrivenLearning.train!(model, X, Y, options)  # warm up
ApplicationDrivenLearning.reset_timer!()
ApplicationDrivenLearning.train!(model, X, Y, options)  # measure
ApplicationDrivenLearning.print_timer()
```
"""
enable_timing!() = TimerOutputs.enable_debug_timings(ApplicationDrivenLearning)

"""
    disable_timing!()

Turn the internal `@timeit_debug` sections back off. See [`enable_timing!`](@ref).
"""
disable_timing!() =
    TimerOutputs.disable_debug_timings(ApplicationDrivenLearning)

"""
    reset_timer!()

Clear every section recorded in [`TIMER`](@ref).
"""
reset_timer!() = TimerOutputs.reset_timer!(TIMER)

"""
    timer()

Return the [`TIMER`](@ref) object, for programmatic inspection with the
`TimerOutputs` API.
"""
timer() = TIMER

"""
    print_timer(io=stdout; kwargs...)

Print the [`TIMER`](@ref) table. Keyword arguments are forwarded to
`TimerOutputs.print_timer`.
"""
print_timer(io::IO = stdout; kwargs...) =
    TimerOutputs.print_timer(io, TIMER; kwargs...)
