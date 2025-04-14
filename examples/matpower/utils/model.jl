ADL = ApplicationDrivenLearning

# build model
model = ADL.Model()

@variable(model, gen[1:pd.n_generators], lower_bound=0.0, ADL.Policy)
@variable(model, rup[1:pd.n_generators], lower_bound=0.0, ADL.Policy)
@variable(model, rdn[1:pd.n_generators], lower_bound=0.0, ADL.Policy)

@variable(model, demand[1:pd.n_demand], ADL.Forecast)
@variable(model, res_up[1:pd.n_zones], ADL.Forecast)
@variable(model, res_dn[1:pd.n_zones], ADL.Forecast)

# schedule (plan) model

# variables
@variables(ADL.Plan(model), begin
    shed[1:pd.n_buses] .>= 0.0
    spil[1:pd.n_buses] .>= 0.0
    ru_shed[1:pd.n_zones] .>= 0.0
    rd_shed[1:pd.n_zones] .>= 0.0
    ru_spil[1:pd.n_zones] .>= 0.0
    rd_spil[1:pd.n_zones] .>= 0.0
    f[1:pd.n_lines]
    θ[1:pd.n_buses]
end)

# basic constraints
@constraints(ADL.Plan(model), begin
    con_max_rup[i=1:pd.n_generators], rup[i].plan <= pd.max_r_up[i]
    con_max_rdn[i=1:pd.n_generators], rdn[i].plan <= pd.max_r_dn[i]
    con_min_f[i=1:pd.n_lines], f[i] >= -pd.F[i]
    con_max_f[i=1:pd.n_lines], f[i] <= pd.F[i]
    con_tension_angles[i=1:pd.n_lines], f[i] == (θ[pd.from[i]] - θ[pd.to[i]]) / pd.x[i]
    con_max_rd[i=1:pd.n_generators], gen[i].plan - rdn[i].plan >= 0.0
    con_max_ru[i=1:pd.n_generators], gen[i].plan + rup[i].plan <= pd.G[i]
end)

# reserve up and down constraints
# TODO: soma de variáveis deve ser feita em `@expression`
for z=1:pd.n_zones
    zone_rup = (size(pd.gen_of_zone[z], 1) > 0 ? sum(rup[i].plan for i in pd.gen_of_zone[z]) : 0.0)
    @constraint(
        ADL.Plan(model), 
        zone_rup + ru_shed[z] - ru_spil[z] == res_up[z].plan,
        base_name="con_reserve_up_zone_$z"
    )

    zone_rdn = (size(pd.gen_of_zone[z], 1) > 0 ? sum(rdn[i].plan for i in pd.gen_of_zone[z]) : 0.0)
    @constraint(
        ADL.Plan(model), 
        zone_rdn + rd_shed[z] - rd_spil[z] == res_dn[z].plan,
        base_name="con_reserve_down_zone_$z"
    )
end

# bus demand balance
for b=1:pd.n_buses
    lod = pd.bus_to_load[b]
    bus_dem = (lod > 0 ? pd.load_scaling[lod] * demand[pd.load_to_demand[lod]].plan : 0.0)
    bus_gen = (size(pd.gen_of_bus[b], 1) > 0 ? sum(gen[i].plan for i in pd.gen_of_bus[b]) : 0.0)
    bus_f1 = (size(pd.line_from_bus[b], 1) > 0 ? sum(f[i] for i in pd.line_from_bus[b]) : 0.0)
    bus_f2 = (size(pd.line_to_bus[b], 1) > 0 ? sum(f[i] for i in pd.line_to_bus[b]) : 0.0)
    bus_f = bus_f1 - bus_f2
    @constraint(
        ADL.Plan(model), 
        bus_gen + shed[b] - spil[b] + bus_f == bus_dem,
        base_name="con_demand_balance_bus_$b"
    )
end

# objective function
cost_gen = sum(pd.c[i] * gen[i].plan + pd.c_r_up[i] * rup[i].plan + pd.c_r_dn[i] * rdn[i].plan for i=1:pd.n_generators)
cost_def = sum(pd.c_deficit * shed[b] for b=1:pd.n_buses)
cost_spl = sum(pd.c_spill * spil[b] for b=1:pd.n_buses)
cost_def_up = sum(pd.c_deficit * 1.1 * ru_shed[z] for z=1:pd.n_zones)
cost_def_dn = sum(pd.c_deficit * 1.1 * rd_shed[z] for z=1:pd.n_zones)
cost_spl_up = sum(pd.c_spill * 1.1 * ru_spil[z] for z=1:pd.n_zones)
cost_spl_dn = sum(pd.c_spill * 1.1 * rd_spil[z] for z=1:pd.n_zones)
cost_final = cost_gen + cost_def + cost_spl + cost_def_up + cost_def_dn + cost_spl_up + cost_spl_dn
@objective(ADL.Plan(model), Min, cost_final)

# dispatch (assess) model

# variables
@variables(ADL.Assess(model), begin
    gen_real_time[1:pd.n_generators]
    shed[1:pd.n_buses] .>= 0.0
    spil[1:pd.n_buses] .>= 0.0
    f[1:pd.n_lines]
    θ[1:pd.n_buses]
end)

# basic constraints
@constraints(ADL.Assess(model), begin
    con_min_f[i=1:pd.n_lines], f[i] >= -pd.F[i]
    con_max_f[i=1:pd.n_lines], f[i] <= pd.F[i]
    con_tension_angles[i=1:pd.n_lines], f[i] == (θ[pd.from[i]] - θ[pd.to[i]]) / pd.x[i]
    con_dn_gen_real_time[i=1:pd.n_generators], gen_real_time[i] >= gen[i].assess - rdn[i].assess
    con_up_gen_real_time[i=1:pd.n_generators], gen_real_time[i] <= gen[i].assess + rup[i].assess
    # con_max_rd[i=1:pd.n_generators], gen[i].assess - rdn[i].assess >= 0.0
    # con_max_ru[i=1:pd.n_generators], gen[i].assess + rup[i].assess <= pd.G[i]
end)

# bus demand balance
for b=1:pd.n_buses
    lod = pd.bus_to_load[b]
    bus_dem = (lod > 0 ? pd.load_scaling[lod] * demand[pd.load_to_demand[lod]].assess : 0.0)
    bus_gen = (size(pd.gen_of_bus[b], 1) > 0 ? sum(gen_real_time[i] for i in pd.gen_of_bus[b]) : 0.0)
    bus_f1 = (size(pd.line_from_bus[b], 1) > 0 ? sum(f[i] for i in pd.line_from_bus[b]) : 0.0)
    bus_f2 = (size(pd.line_to_bus[b], 1) > 0 ? sum(f[i] for i in pd.line_to_bus[b]) : 0.0)
    bus_f = bus_f1 - bus_f2
    @constraint(
        ADL.Assess(model), 
        bus_gen + shed[b] - spil[b] + bus_f == bus_dem,
        base_name="con_demand_balance_bus_$b"
    )
end

# objective function
cost_gen = sum(pd.c[i] * gen_real_time[i] + pd.c_r_up[i] * rup[i].assess + pd.c_r_dn[i] * rdn[i].assess for i=1:pd.n_generators)
cost_def = sum(pd.c_deficit * shed[b] for b=1:pd.n_buses)
cost_spl = sum(pd.c_spill * spil[b] for b=1:pd.n_buses)
cost_final = cost_gen + cost_def + cost_spl
@objective(ADL.Assess(model), Min, cost_final)

set_optimizer(model, Gurobi.Optimizer)
set_silent(model)
ADL.build(model)
