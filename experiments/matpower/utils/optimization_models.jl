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

Optimization model definition.
Scheduling and Dispatch models are defined as JuMP models.
=#

function zone_demand(pd::PhysicalData, d, zone, t)
    # load struct fields into local scope
    @unpack_PhysicalData pd
    out = 0.0
    for dem = 1:n_demand
        if demand_to_zone[dem] == zone
            out += d[dem, t]
        end
    end
    return out
end

# Build Gp
# Planning model
# Generation and reserve scheduling
function build_schedule(pd::PhysicalData, model, θd, θru, θrd, d)
    # load struct fields into local scope
    @unpack_PhysicalData pd

    T = lastindex(d, 2)

    g = [@variable(model[t], lower_bound = 0.0) for i = 1:n_generators, t = 1:T] #=, base_name="g_$(i)_$t"=#
    ru = [
        @variable(model[t], lower_bound = 0.0, upper_bound = max_r_up[i]) for
        i = 1:n_generators, t = 1:T
    ] #=, base_name="ru_$(i)_$t"=#
    rd = [
        @variable(model[t], lower_bound = 0.0, upper_bound = max_r_dn[i]) for
        i = 1:n_generators, t = 1:T
    ] #=, base_name="rd_$(i)_$t"=#

    shed = [@variable(model[t], lower_bound = 0.0) for i = 1:n_buses, t = 1:T] #=, base_name="sh_$(i)_$t"=#
    spil = [@variable(model[t], lower_bound = 0.0) for i = 1:n_buses, t = 1:T] #=, base_name="sp_$(i)_$t"=#

    ru_shed = [@variable(model[t], lower_bound = 0.0) for i = 1:n_zones, t = 1:T] #=, base_name="rus_$(i)_$t"=#
    rd_shed = [@variable(model[t], lower_bound = 0.0) for i = 1:n_zones, t = 1:T] #=, base_name="rds_$(i)_$t"=#
    ru_spil = [@variable(model[t], lower_bound = 0.0) for i = 1:n_zones, t = 1:T] #=, base_name="rus_$(i)_$t"=#
    rd_spil = [@variable(model[t], lower_bound = 0.0) for i = 1:n_zones, t = 1:T] #=, base_name="rds_$(i)_$t"=#

    f = [
        @variable(model[t], lower_bound = -F[i], upper_bound = F[i]) for i = 1:n_lines,
        t = 1:T
    ]

    # tension angles (do not confuse with AR parameters)
    θ = [@variable(model[t]) for i = 1:n_buses, t = 1:T]
    for t = 1:T, i = 1:n_lines
        @constraint(model[t], f[i, t] == (θ[from[i], t] - θ[to[i], t]) / x[i])
    end

    for t = 1:T, i = 1:n_generators
        @constraint(model[t], g[i, t] - rd[i, t] >= 0)
        @constraint(model[t], g[i, t] + ru[i, t] <= G[i])
    end

    if θd === nothing
        for t = 1:T, b = 1:n_buses
            lod = bus_to_load[b]
            @constraint(
                model[t],
                sum(g[i, t] for i in gen_of_bus[b]) + shed[b, t] - spil[b, t] +
                sum(f[i, t] for i in line_from_bus[b]) -
                sum(f[i, t] for i in line_to_bus[b]) ==
                (lod > 0 ? load_scaling[lod] * d[load_to_demand[lod], t] : 0.0)
            )
        end
    else
        for t = 1:T, b = 1:n_buses
            lod = bus_to_load[b]
            @constraint(
                model[t],
                sum(g[i, t] for i in gen_of_bus[b]) + shed[b, t] - spil[b, t] +
                sum(f[i, t] for i in line_from_bus[b]) -
                sum(f[i, t] for i in line_to_bus[b]) == sum(
                    θd[load_to_demand[lod], p][t] *
                    load_scaling[lod] *
                    ifelse(p == 0, 1, d[load_to_demand[lod], t-p]) for
                    p in axes(θd, 2) if lod > 0
                )
            )
        end
    end

    if θru !== nothing
        for t = 1:T, b = 1:n_zones
            @constraint(
                model[t],
                sum(ru[i, t] for i in gen_of_zone[b]) + ru_shed[b, t] - ru_spil[b, t] ==
                sum(
                    θru[b, p][t] * ifelse(p == 0, 1, zone_demand(pd, d, b, t - p))
                    for p in axes(θru, 2)
                )
            )
        end
    end

    if θrd !== nothing
        for t = 1:T, b = 1:n_zones
            @constraint(
                model[t],
                sum(rd[i, t] for i in gen_of_zone[b]) + rd_shed[b, t] - rd_spil[b, t] ==
                sum(
                    θru[b, p][t] * ifelse(p == 0, 1, zone_demand(pd, d, b, t - p))
                    for p in axes(θru, 2)
                )
            )
        end
    end

    for batch=1:T  # batch_list(model)
        @objective(
            model[batch],
            Min,
            sum(
                sum(
                    c[i] * g[i, t] + c_r_up[i] * ru[i, t] + c_r_dn[i] * rd[i, t] for
                    i = 1:n_generators
                ) +
                sum(c_deficit * shed[b, t] for b = 1:n_buses) +
                sum(c_spill * spil[b, t] for b = 1:n_buses) +
                sum((c_deficit * 1.1) * ru_shed[b, t] for b = 1:n_zones) +
                sum((c_deficit * 1.1) * rd_shed[b, t] for b = 1:n_zones) +
                sum((c_spill * 1.1) * ru_spil[b, t] for b = 1:n_zones) +
                sum((c_spill * 1.1) * rd_spil[b, t] for b = 1:n_zones) for
                t=1:T
            )
        )
    end

    return g, ru, rd
end

# Build Ga
# Assessment model
# Real time dispatch after uncertainty realization
function build_dispatch(pd::PhysicalData, model, g, ru, rd, d)
    # load struct fields into local scope
    @unpack_PhysicalData pd

    T = lastindex(d, 2)

    # variables will have subscript "d" do differentiate from the scheduling model

    # real time generation
    g_d = [@variable(model[t], lower_bound = 0.0) for i = 1:n_generators, t = 1:T]

    # load shedding
    shed_d = [@variable(model[t], lower_bound = 0.0) for i = 1:n_buses, t = 1:T]
    # load spillage
    spil_d = [@variable(model[t], lower_bound = 0.0) for i = 1:n_buses, t = 1:T]

    # line flows
    f_d = [
        @variable(model[t], lower_bound = -F[i], upper_bound = F[i]) for i = 1:n_lines,
        t = 1:T
    ]

    # tension angles (do not confuse with AR parameters)
    θ_d = [@variable(model[t]) for i = 1:n_buses, t = 1:T]
    for i = 1:n_lines, t = 1:T
        @constraint(model[t], f_d[i, t] == (θ_d[from[i], t] - θ_d[to[i], t]) / x[i])
    end

    # real time generation limits
    for t = 1:T, i = 1:n_generators
        @constraint(model[t], g_d[i, t] >= g[i, t] - rd[i, t])
        @constraint(model[t], g_d[i, t] <= g[i, t] + ru[i, t])
    end

    # demmand balance
    for b = 1:n_buses, t = 1:T
        lod = bus_to_load[b]
        @constraint(
            model[t],
            sum(g_d[i, t] for i in gen_of_bus[b]) + shed_d[b, t] - spil_d[b, t] +
            sum(f_d[i, t] for i in line_from_bus[b]) -
            sum(f_d[i, t] for i in line_to_bus[b]) ==
            (lod > 0 ? load_scaling[lod] * d[load_to_demand[lod], t] : 0.0)
        )
    end

    for batch=1:T  # batch_list(model)
        @objective(
            model[batch],
            Min,
            sum(
                sum(
                    c[i] * (g_d[i, t]) + c_r_up[i] * ru[i, t] + c_r_dn[i] * rd[i, t] for
                    i = 1:n_generators
                ) +
                sum(c_deficit * shed_d[b, t] for b = 1:n_buses) +
                sum(c_spill * spil_d[b, t] for b = 1:n_buses) for t=1:T
            )
        )
    end

    return g_d, shed_d, spil_d
end

function simulate_in(pd::PhysicalData, θd, θru, θrd, d, name::String = "")::Float64
    T = lastindex(d, 2)
    sched = [JuMP.Model(HiGHS.Optimizer) for t=1:T]  # BatchedModel(simulation_opt(), T)
    set_silent.(sched)
    _g, _ru, _rd = build_schedule(pd, sched, θd, θru, θrd, d)
    optimize!.(sched)

    stat = termination_status(sched[1])
    if stat != MOI.OPTIMAL
        println(FILE, "sim schedule: termination status was $(stat) at $(name)")
        error("sim schedule: termination status was $(stat) at $(name)")
        return Inf
    end

    g = value.(_g)
    ru = value.(_ru)
    rd = value.(_rd)

    T = lastindex(d, 2)
    disp = [JuMP.Model(HiGHS.Optimizer) for t=1:T]  # BatchedModel(simulation_opt(), T)
    set_silent.(disp)
    build_dispatch(pd, disp, g, ru, rd, d)
    optimize!.(disp)

    stat = termination_status(disp[1])
    if stat != MOI.OPTIMAL
        println(FILE, "sim dispatch: termination status was $(stat) at $(name)")
        error("sim dispatch: termination status was $(stat) at $(name)")
        return Inf
    end

    T = lastindex(d, 2)
    obj = sum(objective_value.(disp)) / T

    return obj
end

function simulate(
    pd::PhysicalData,
    θd,
    θru,
    θrd,
    d,
    n_stage_groups,
    name::String = "",
)::Float64
    max_stage_groups = min(n_stage_groups, lastindex(d, 2))
    # separate out-of-sample load into blocks to speedup optimization problems
    demands_vector =
        [split_demand_pmap(pd, d, step, max_stage_groups) for step = 1:max_stage_groups]
    obj = [simulate_in(pd, θd, θru, θrd, _demand, name) for _demand in demands_vector]
    # @time obj =
    #     pmap(_demand -> simulate_in(pd, θd, θru, θrd, _demand, name), demands_vector)
    return Statistics.mean(obj)
end

function split_demand_pmap(pd::PhysicalData, d, group, max_stage_groups)

    @unpack_PhysicalData pd

    T = lastindex(d, 2) # samples / stages
    n_dem = lastindex(d, 1) # individual demands

    i = group

    min_t_per_core, remaining = divrem(T, max_stage_groups)

    # number of stages/samples to be simulated at this call
    current_T = min_t_per_core + ifelse(i <= remaining, 1, 0)

    current_first = 1 + (i - 1) * min_t_per_core + min(i - 1, remaining)

    max_lags = max(n_demand_lags, n_reserve_lags)

    current_d = OffsetArray{Float64}(undef, 1:n_dem, (1-max_lags):current_T)

    # move part of the mais array to a local one
    copyto!(
        current_d.parent,
        d[:, (current_first-max_lags):(current_first-1+current_T)].parent,
    )

    return current_d
end
