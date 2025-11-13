######################################################################
#  Copyright 2024, Joaquim Dias Garcia, and contributors
#  This Source Code Form is subject to the terms of the Mozilla Public
#  License, v. 2.0. If a copy of the MPL was not distributed with this
#  file, You can obtain one at https://mozilla.org/MPL/2.0/.
######################################################################
# This file is part of the publication:
#  Application-Driven Learning: A Closed-Loop
#  Prediction and Optimization Approach Applied to
#  Dynamic Reserves and Demand Forecasting
# by:
#  Joaquim Dias Garcia, Alexandre Street,
#  Tito Homem-de-Mello & Francisco D. Munoz
######################################################################
# This code base is part of an academic work and should not be
# considered ready for production.
######################################################################

#=
Info:

This file implements the heuristic algorithm
the optimization method used to update the
forecast models is Nelder Mead implementes in Optim.jl.
The Nelder-Mead main steps are only executed in the main process
the call to linear programming solvers to optimize Ga and Gp
are parallel inthe samples.
=#

mutable struct OptimData{M}
    sched_model::M
    disp_model::M

    var_g::Array{JuMP.VariableRef,2}
    var_ru::Array{JuMP.VariableRef,2}
    var_rd::Array{JuMP.VariableRef,2}

    param_g::Array{ParameterJuMP.ParameterRef,2}
    param_ru::Array{ParameterJuMP.ParameterRef,2}
    param_rd::Array{ParameterJuMP.ParameterRef,2}

    param_θd::Union{Nothing,OffsetArray{BatchParameterRef,2,Array{BatchParameterRef,2}}}
    param_θru::Union{Nothing,OffsetArray{BatchParameterRef,2,Array{BatchParameterRef,2}}}
    param_θrd::Union{Nothing,OffsetArray{BatchParameterRef,2,Array{BatchParameterRef,2}}}

    # objective (obj) and solution (x)
    # vector since its the current solution in each parallel process
    obj_all::Vector{Float64}
    x_all::Vector{Vector{Float64}}

    T::Int

    # shared objects available to all processes / workers
    X_SHARED::SharedArrays.SharedArray{Float64}
    OBJ_SHARED::SharedArrays.SharedArray{Float64}
    function OptimData(sched::M, disp::M) where {M}
        dt = new{M}(sched, disp)
        return dt
    end
end

num_params(d::OffsetArray{BatchParameterRef}) = length(d)
num_params(d) = 0

not_nothing(a::AbstractArray, b::Nothing) = a
not_nothing(a::Nothing, b::AbstractArray) = b
not_nothing(a::Nothing, b::Nothing) = error("both nothing")

# similar to split_demand_pmap
# but this splits the training data evenly among processes
function split_demand(pd::PhysicalData, d)

    @unpack_PhysicalData pd

    T = lastindex(d, 2)
    n_dem = lastindex(d, 1)

    N = nprocs()

    i = myid() # current process

    min_t_per_core, remaining = divrem(T, N)

    # remaining < N hence this is correct - different from batch model

    my_T = min_t_per_core + ifelse(i <= remaining, 1, 0)

    my_first = 1 + (i - 1) * min_t_per_core + min(i - 1, remaining)

    max_lags = max(n_demand_lags, n_reserve_lags)

    my_d = OffsetArray{Float64}(undef, 1:n_dem, (1-max_lags):my_T)

    copyto!(my_d.parent, d[:, (my_first-max_lags):(my_first-1+my_T)].parent)

    return my_d
end

# function called in all processes
function optim_init(
    pd::PhysicalData,
    dd,
    in_θd,
    in_θru,
    in_θrd,
)

    tot_T = lastindex(dd, 2)

    # make struct fields availabel in current scope
    @unpack_PhysicalData pd

    # we need to select slices of d - with offsets depending on myid()
    d = split_demand(pd, dd)

    T = lastindex(d, 2)

    # sched = BatchedModel(optim_opt(), T, true)
    # set_silent(sched)

    # disp = BatchedModel(optim_opt(), T, true)
    # set_silent(disp)

    model = ApplicationDrivenLearning.Model()

    g, ru, rd = build_schedule(pd, sched, θd, θru, θrd, d)
    dt.var_g = g
    dt.var_ru = ru
    dt.var_rd = rd

    dt.param_g = similar(g, ParameterJuMP.ParameterRef, size(g))
    for i = 1:n_generators, t = 1:T
        dt.param_g[i, t] = ParameterJuMP._add_parameter(disp[t], 0.0)
    end
    dt.param_ru = similar(ru, ParameterJuMP.ParameterRef, size(ru))
    for i = 1:n_generators, t = 1:T
        dt.param_ru[i, t] = ParameterJuMP._add_parameter(disp[t], 0.0)
    end
    dt.param_rd = similar(rd, ParameterJuMP.ParameterRef, size(rd))
    for i = 1:n_generators, t = 1:T
        dt.param_rd[i, t] = ParameterJuMP._add_parameter(disp[t], 0.0)
    end

    T = lastindex(d, 2)

    _ru = [@variable(disp[t]) for i = 1:n_generators, t = 1:T]
    _rd = [@variable(disp[t]) for i = 1:n_generators, t = 1:T]

    for t = 1:T, i = 1:n_generators
        @constraint(disp[t], _ru[i, t] == dt.param_ru[i, t])
        @constraint(disp[t], _rd[i, t] == dt.param_rd[i, t])
    end

    build_dispatch(pd, disp, dt.param_g, _ru, _rd, d)

    # retur initialized data
    return dt
end

function optim_solve(
    pd::PhysicalData,
    time_limit,
    d;
    in_θd = nothing,
    in_θru = nothing,
    in_θrd = nothing,
    start_θd = nothing,
    start_θru = nothing,
    start_θrd = nothing,
)

    # initialize the main struct in all processes
    dt = optim_init(pd, d, in_θd, in_θru, in_θrd)

    # stack of forecast model parameters for optimization
    x0 =
        zeros(num_params(dt.param_θd) + num_params(dt.param_θru) + num_params(dt.param_θrd))
    load_x!(dt, x0, start_θd, start_θru, start_θrd)

    # initialize shared objects
    dt.X_SHARED = SharedArrays.SharedArray{Float64}(length(x0))
    dt.OBJ_SHARED = SharedArrays.SharedArray{Float64}(nprocs())
    @everywhere workers() dt.X_SHARED = $(dt.X_SHARED)
    @everywhere workers() dt.OBJ_SHARED = $(dt.OBJ_SHARED)
    copyto!(dt.X_SHARED, x0)

    res = Optim.optimize(
        x -> min_costx(x, dt),
        x0;
        inplace = true,
        f_tol = 1e-6,
        x_tol = 1e-5,
        g_tol = 1e-6,
        iterations = 30 * 1_000,
        time_limit = time_limit,
        show_trace = true,
        show_every = 10, #50 #250, #for 1 bus
    )

    # query and move final solution
    xf = Optim.minimizer(res)
    unload_x!(dt, xf, start_θd, start_θru, start_θrd)
    x_all = deepcopy(dt.x_all)
    obj_all = deepcopy(dt.obj_all)

    # cleanup memory in all processes
    @everywhere begin # sync
        dt = nothing
        GC.gc()
    end

    return not_nothing(in_θd, start_θd),
    not_nothing(in_θru, start_θru),
    not_nothing(in_θrd, start_θrd),
    x_all,
    obj_all
end

# move nelder mead new candidate to the forecast model
# of the linear programs
function set_θ!(dt::OptimData{M}, x) where {M}
    n = num_params(dt.param_θd)
    if n > 0
        for i = 1:n
            set_value(dt.param_θd.parent[i], x[i])
        end
    end
    n = num_params(dt.param_θru)
    if n > 0
        shift = num_params(dt.param_θd)
        for i = 1:n
            set_value(dt.param_θru.parent[i], x[i+shift])
        end
    end
    n = num_params(dt.param_θrd)
    if n > 0
        shift = num_params(dt.param_θd) + num_params(dt.param_θru)
        for i = 1:n
            set_value(dt.param_θrd.parent[i], x[i+shift])
        end
    end
    return nothing
end

# initialize nelder mead solution vector
function load_x!(dt::OptimData{M}, x, θd, θru, θrd) where {M}
    n = num_params(dt.param_θd)
    if n > 0
        for i = 1:n
            x[i] = θd.parent[i]
        end
    end
    n = num_params(dt.param_θru)
    if n > 0
        shift = num_params(dt.param_θd)
        for i = 1:n
            x[i+shift] = θru.parent[i]
        end
    end
    n = num_params(dt.param_θrd)
    if n > 0
        shift = num_params(dt.param_θd) + num_params(dt.param_θru)
        for i = 1:n
            x[i+shift] = θrd.parent[i]
        end
    end
    return nothing
end

# move nelder mead solution to individual vectors
function unload_x!(dt::OptimData{M}, x, θd, θru, θrd) where {M}
    n = num_params(dt.param_θd)
    if n > 0
        for i = 1:n
            θd.parent[i] = x[i]
        end
    end
    n = num_params(dt.param_θru)
    if n > 0
        shift = num_params(dt.param_θd)
        for i = 1:n
            θru.parent[i] = x[i+shift]
        end
    end
    n = num_params(dt.param_θrd)
    if n > 0
        shift = num_params(dt.param_θd) + num_params(dt.param_θru)
        for i = 1:n
            θrd.parent[i] = x[i+shift]
        end
    end
    return nothing
end

# move solution from scheduling (Gp) to dispatch (Ga) model
function set_dipatch!(dt::OptimData{M}) where {M}
    for i in eachindex(dt.var_g)
        var = dt.var_g[i]
        if model_ok(var)
            set_value(dt.param_g[i], max(0.0, value(var)))
        end
    end
    for i in eachindex(dt.var_ru)
        var = dt.var_ru[i]
        if model_ok(var)
            set_value(dt.param_ru[i], max(0.0, value(var)))
        end
    end
    for i in eachindex(dt.var_rd)
        var = dt.var_rd[i]
        if model_ok(var)
            set_value(dt.param_rd[i], max(0.0, value(var)))
        end
    end
    return nothing
end

# nelder meads "blackbox function"
function min_costx(x, dt::OptimData{M}) where {M}
    # make current solution candidate availabel in all processes
    copyto!(dt.X_SHARED, x)
    # optimize the multiple samples in parallel
    @everywhere begin # sync
        parallel_opt(dt::OptimData{BatchedModel{Model}})::Nothing
    end
    val = sum(dt.OBJ_SHARED) / dt.T
    return val
end

# given new solution
# solve schedule (Gp)
# solve dispatch (Ga)
# query cost
function parallel_opt(dt::OptimData{M}) where {M}
    set_θ!(dt, dt.X_SHARED::SharedArrays.SharedArray{Float64,1})::Nothing
    JuMP.optimize!(dt.sched_model::BatchedModel)::Nothing
    set_dipatch!(dt)::Nothing
    JuMP.optimize!(dt.disp_model::M)::Nothing
    dt.OBJ_SHARED[myid()] = objective_value(dt.disp_model::M)::Float64
    stat = termination_status(dt.sched_model)
    if stat != MOI.OPTIMAL
        error("Failed at scheduling $stat, $(myid())")
    end
    stat = termination_status(dt.disp_model)
    if stat != MOI.OPTIMAL
        error("Failed at dispatch $stat, $(myid())")
    end
    return nothing
end
