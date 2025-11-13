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

This file is responsible for converting raw data from matpower files
into a self contained "PhysicalData" struct tha holds all the
systems physical data.
Part of the data that is not in the matpower file is created here
given parameters defined inthe configurations file.
=#

Parameters.@with_kw mutable struct PhysicalData

    # number of elements
    #
    # number of generators
    n_generators::Int = 0
    # number of lines
    n_lines::Int = 0
    # number of buses
    n_buses::Int = 0
    # number of demands
    # might be smaller than number of buses
    n_demand::Int = 0
    # number of reserve zones
    n_zones::Int = 1

    # generator data
    #
    # Gen. capacity
    G::Vector{Float64} = Float64[]
    # Gen. costs
    c::Vector{Float64} = Float64[]
    # bus from the i-th generator
    generator_to_bus::Vector{Int} = Int[]

    # reserves data
    #
    # cost of up reserves
    c_r_up::Vector{Float64} = Float64[]
    # cost of down reserves
    c_r_dn::Vector{Float64} = Float64[]
    # maximum up reserve per generator
    max_r_up::Vector{Float64} = Float64[]
    # maximum down reserve per generator
    max_r_dn::Vector{Float64} = Float64[]
    # (reserve) zone from bus i
    bus_to_zone::Vector{Int} = Int[]

    # network data
    #
    # maximum line flow
    F::Vector{Float64} = Float64[]
    # line reactance
    x::Vector{Float64} = Float64[]
    # `from` bus of line `i`
    from::Vector{Int} = Int[]
    # `to` bus of line i
    to::Vector{Int} = Int[]

    # penalties
    #
    # load spillage cost
    c_spill::Float64 = 0.0
    # load shedding cost
    c_deficit::Float64 = 0.0

    # forecast model size
    #
    # AR lags for load forecast
    n_demand_lags::Int = 0
    # AR lags for reserve model
    n_reserve_lags::Int = 0

    # energy consumption
    #
    # demand data
    # demand is an *agreggation* of loads in separate buses.
    # In other words: sum of loads of a group of buses
    # this represents the fact that the load in some buses vary jointly
    # each bus (in this demand group) will have a share of this demand.
    # Time series will be attached to demands,
    # bus loads that compose the same demand will have scaled versions
    # of the same time series.
    demand_value::Vector{Float64} = Float64[]
    # obtain the zone of a demand
    # each demand is split among buses in the same zone
    demand_to_zone::Vector{Int} = Int[]
    # loads data
    # the load scaling: a multiple (share) of the underlying demand
    # i.e. load_scaling * base_demand = bus_demand
    load_scaling::Vector{Float64} = Float64[]
    # obtain the undelying demand index from a load index
    load_to_demand::Vector{Int} = Int[]
    # obtain the load index from a bus
    bus_to_load::Vector{Int} = Int[]

    # maps for faster model building
    #
    # all lines `from` bus i
    line_from_bus::Vector{Vector{Int}} = Vector{Int}[]
    # all lines `to` bus i
    line_to_bus::Vector{Vector{Int}} = Vector{Int}[]
    # all generators in bus i
    gen_of_bus::Vector{Vector{Int}} = Vector{Int}[]
    # all generators in zone i
    gen_of_zone::Vector{Vector{Int}} = Vector{Int}[]

end

Parameters.@with_kw mutable struct TimeSeriesGenerator
    stages_train::Int = 0
    stages_test::Int = 0
    buffer::Int = 0

    # std / mean
    coef_of_variation::Float64 = 0.0

    θd_lag::Vector{Float64} = Float64[] # only AR 1
end

function convert_matpower(
    mpc::Matpower,
    N_LAGS::Int,
    N_DEMANDS::Int,
    N_ZONES::Int,
    TEST_SIZE::Int,
    DEFF_COEF::Float64,
    SPILL_COEF::Float64,
    COEF_VARIATION::Float64,
    DEMAND_FILE::String,
    train_dataset_size_current::Int,
    sample::Int = 1,
)

    #
    # Bus raw data
    #

    # cache relation from a bus to its matpower `code`
    bus2code = map(z -> z.bus_i, mpc.bus)
    # compute reverse relation
    code2bus = Dict{Int,Int}()
    for (bus, code) in enumerate(bus2code)
        code2bus[code] = bus
    end
    _code_to_bus(code::Integer) = code2bus[code]
    raw_demand_per_bus = map(z -> z.Pd, mpc.bus) # real power

    #
    # Generator raw data
    #

    gen_bus = map(z -> z.bus, mpc.gen)#Int.(mpc.gen[:,1])
    gen_cap = map(z -> z.Pmax, mpc.gen)#mpc.gen[:,9] # real power
    # gen_cost_quad = map(z->z.c2, mpc.gencost)#mpc.costcost[:,5] # quad coef
    gen_cost_lin = map(z -> z.c1, mpc.gencost)#mpc.gencost[:,6] # lin coef
    # gen_cost_cte = mpc.gencost[:,7] # cte coef
    gen_cost = 0.1 * (gen_cost_lin)

    # remove generators with capacity 0 (or negative)
    valid_gen = Int[]
    for i in eachindex(gen_cap)
        if gen_cap[i] > 0
            push!(valid_gen, i)
        end
    end
    gen_bus = gen_bus[valid_gen]
    gen_cap = gen_cap[valid_gen]
    gen_cost = gen_cost[valid_gen]

    #
    # Line raw data
    #

    if length(mpc.branch) > 0
        line_from = map(z -> z.fbus, mpc.branch)#Int.(mpc.branch[:, 1])
        line_to = map(z -> z.tbus, mpc.branch)#Int.(mpc.branch[:, 2])
        # line_r = map(z->z.r, mpc.branch)# mpc.branch[:, 3]
        line_x = map(z -> z.x, mpc.branch)# mpc.branch[:, 4]
        # line_b = map(z->z.b, mpc.branch)# mpc.branch[:, 5]
        line_cap = map(z -> z.rateA, mpc.branch)# mpc.branch[:, 6] # zero is infinity
    end

    # cache data into main struct
    ps_data = PhysicalData()
    ps_data.n_generators = length(gen_bus)
    ps_data.G = gen_cap
    ps_data.c = gen_cost
    ps_data.generator_to_bus = _code_to_bus.(gen_bus)
    if length(mpc.branch) > 0
        ps_data.n_lines = length(line_cap)
        ps_data.F = line_cap
        ps_data.x = line_x
        ps_data.from = _code_to_bus.(line_from)
        ps_data.to = _code_to_bus.(line_to)
    end
    ps_data.n_buses = length(raw_demand_per_bus)

    #
    # Reserves data
    #

    # create reserve data
    ps_data.c_r_up = 0.3 .* ps_data.c
    ps_data.c_r_dn = 0.3 .* ps_data.c
    ps_data.max_r_up = 0.3 .* ps_data.G
    ps_data.max_r_dn = 0.3 .* ps_data.G

    #
    # Zones data
    #

    if N_ZONES > 0
        # creates zones by splitting a network
        ps_data.bus_to_zone = split_net(ps_data.from, ps_data.to, ps_data.n_buses, N_ZONES)
    else
        # uses original zones from .m file
        ps_data.bus_to_zone = map(z -> z.area, mpc.bus) # Int.(mpc.bus[:,7])
    end
    ps_data.n_zones = maximum(ps_data.bus_to_zone)

    #
    # Load raw data
    #

    # create temporary element `load` which is a "bus with demand"
    _n_load = count(x -> x > 0, raw_demand_per_bus)

    _demand_per_load = zeros(_n_load)
    _bus_to_load = zeros(Int, ps_data.n_buses)
    _load_to_bus = zeros(Int, _n_load)
    c = 0
    for b = 1:ps_data.n_buses
        if raw_demand_per_bus[b] > 0
            c += 1
            _bus_to_load[b] = c
            _load_to_bus[c] = b
            _demand_per_load[c] = raw_demand_per_bus[b]
        end
    end
    @assert c == _n_load

    #
    # Demand data
    #

    if N_DEMANDS > 0
        # agreggate loads into fewer demands
        if N_ZONES > N_DEMANDS
            error(
                "N_ZONES must be smaller than N_DEMANDS, got N_ZONES = $(N_ZONES) and N_DEMANDS = $(N_DEMANDS)",
            )
        end

        max_dem_per_zone = max(div(N_DEMANDS, ps_data.n_zones), 1)

        # note: a demand can only have Loads in a single zone

        dem_idx_in_zone_to_dem_idx =
            [[0 for _ = 1:max_dem_per_zone] for _ = 1:ps_data.n_zones]

        loads_in_zone_count = [0 for _ = 1:ps_data.n_zones]

        _n_demand = 0
        _demand_value = Float64[] # energy per demand (sum of loads)

        load_to_demand = zeros(Int, _n_load)
        demand_to_zone = Int[]
        # load as a fraction of the overall demand
        _load_scaling = zeros(Float64, _n_load)

        for z = 1:ps_data.n_zones
            for l = 1:_n_load
                if ps_data.bus_to_zone[_load_to_bus[l]] == z
                    # increase number of loads in this zone so far
                    loads_in_zone_count[z] += 1
                    # alternatingly add load to a demand group index IN THE ZONE
                    dem_idx_in_zone = mod1(loads_in_zone_count[z], max_dem_per_zone)
                    # get the global demand index (of the demand group in the zone)
                    dem_idx = dem_idx_in_zone_to_dem_idx[z][dem_idx_in_zone]
                    if dem_idx == 0
                        # if zero, creates a global demand
                        _n_demand += 1
                        # update index
                        dem_idx = _n_demand
                        # save index
                        dem_idx_in_zone_to_dem_idx[z][dem_idx_in_zone] = dem_idx
                    end
                    # cache map
                    load_to_demand[l] = dem_idx
                    if dem_idx > length(_demand_value)
                        # add data of new demand
                        push!(_demand_value, _demand_per_load[l])
                        push!(demand_to_zone, z)
                        @assert dem_idx == length(_demand_value)
                    else
                        # update data of existing demand
                        # i.e. add load value to demand value
                        _demand_value[dem_idx] += _demand_per_load[l]
                    end
                end
            end
        end
        # compute load fractional share of a demand
        for l = 1:_n_load
            d = load_to_demand[l]
            _load_scaling[l] = _demand_per_load[l] / _demand_value[d]
        end

        ps_data.load_to_demand = load_to_demand
        ps_data.load_scaling = _load_scaling
        ps_data.demand_to_zone = demand_to_zone
        _load_to_bus = deepcopy(_load_to_bus)
        ps_data.bus_to_load = zeros(Int, ps_data.n_buses)
        for (k, v) in enumerate(_load_to_bus)
            ps_data.bus_to_load[v] = k
        end

        ps_data.demand_value = _demand_value # sum within loads
        ps_data.n_demand = _n_demand # reduced number
    else
        # 1 to 1 mapping between loads and demands
        ps_data.n_demand = _n_load
        ps_data.demand_value = _demand_per_load
        ps_data.load_to_demand = collect(1:ps_data.n_demand)
        ps_data.load_scaling = ones(ps_data.n_demand)
        _load_to_bus = deepcopy(_load_to_bus)
        ps_data.bus_to_load = deepcopy(_bus_to_load)
        ps_data.demand_to_zone = zeros(Int, ps_data.n_demand)
        for dem = 1:ps_data.n_demand
            ps_data.demand_to_zone[dem] = ps_data.bus_to_zone[_load_to_bus[dem]]
        end
    end

    #
    # additional maps for faster model
    #

    ps_data.line_from_bus = [Int[] for _ = 1:ps_data.n_buses]
    ps_data.line_to_bus = [Int[] for _ = 1:ps_data.n_buses]
    ps_data.gen_of_bus = [Int[] for _ = 1:ps_data.n_buses]
    ps_data.gen_of_zone = [Int[] for _ = 1:ps_data.n_zones]
    for i = 1:ps_data.n_lines
        bus_f = ps_data.from[i]
        push!(ps_data.line_from_bus[bus_f], i)
        bus_t = ps_data.to[i]
        push!(ps_data.line_to_bus[bus_t], i)
    end
    for i = 1:ps_data.n_generators
        bus = ps_data.generator_to_bus[i]
        push!(ps_data.gen_of_bus[bus], i)
    end
    for i = 1:ps_data.n_generators
        bus = ps_data.bus_to_zone[ps_data.generator_to_bus[i]]
        push!(ps_data.gen_of_zone[bus], i)
    end

    #
    # System data
    #

    ps_data.c_spill = SPILL_COEF * maximum(gen_cost)
    ps_data.c_deficit = DEFF_COEF * maximum(gen_cost)

    #
    # Forecast model data
    #

    ps_data.n_demand_lags = N_LAGS
    ps_data.n_reserve_lags = 0

    #
    # Time series data
    #

    tsg = TimeSeriesGenerator()
    tsg.stages_train = train_dataset_size_current
    tsg.stages_test = TEST_SIZE
    tsg.buffer = 100

    tsg.coef_of_variation = COEF_VARIATION
    tsg.θd_lag = 0.9 * ones(ps_data.n_demand)

    ts_data = if !isempty(DEMAND_FILE)
        demand_time_series_file(ps_data, tsg, DEMAND_FILE, sample)
    else
        demand_time_series_synthetic(ps_data, tsg, sample)
    end

    return ps_data, ts_data
end

# a non-random algorithm to split a network into zones
function split_net(edge_to_node_from, edge_to_node_to, n_nodes, n_zones)

    if n_zones > n_nodes
        n_zones = n_nodes
    end
    if n_zones == 1
        return collect(1:n_nodes)
    end

    # a target number
    nodes_per_zone = div(n_nodes, n_zones)

    @assert length(edge_to_node_from) == length(edge_to_node_to)

    n_edges = length(edge_to_node_from)

    # map for each to node to its neighbors
    connection = Dict{Int,Vector{Int}}()
    for i = 1:n_edges
        n1 = edge_to_node_from[i]
        n2 = edge_to_node_to[i]
        if haskey(connection, n1)
            push!(connection[n1], n2)
        else
            connection[n1] = Int[n2]
        end
        if haskey(connection, n2)
            push!(connection[n2], n1)
        else
            connection[n2] = Int[n1]
        end
    end

    zone = zeros(Int, n_nodes)

    zone_size = zeros(Int, n_zones)

    current_zone = 1
    for _ = 1:100_000

        # find a free (without a zone yet) node with degree 1
        selected = 0
        for (node, conn) in connection
            if zone[node] == 0 && length(conn) == 1
                selected = node
            end
        end
        # if there ir no free node of degree 1, get any other free node
        if selected == 0
            for node = 1:n_nodes
                if zone[node] == 0
                    selected = node
                end
            end
        end

        # add node to current zone
        zone[selected] = current_zone
        zone_size[current_zone] += 1

        current_set = Int[selected]
        for _ = 1:20_000
            if zone_size[current_zone] >= nodes_per_zone
                break # zone completed
            end
            current_set = add_nodes(
                current_set,
                zone,
                zone_size,
                current_zone,
                nodes_per_zone,
                connection,
            )
        end
        if zone_size[current_zone] >= nodes_per_zone
            current_zone += 1 # move to next zone
        end
        if current_zone > n_zones
            break # number of zones completed
        end
    end

    # associate missing nodes
    for _ = 1:20_000
        done_something = false
        for node = 1:n_nodes
            if zone[node] == 0
                for nei in connection[node]
                    if zone[nei] != 0
                        zone[node] = zone[nei]
                        zone_size[zone[node]] += 1
                        done_something = true
                        break
                    end
                end
            end
        end
        if !done_something
            break
        end
    end

    # final pass for isolated nodes
    for node = 1:n_nodes
        if zone[node] == 0
            zone[node] = 1
            zone_size[zone[node]] += 1
        end
    end

    return zone
end

function add_nodes(current_set, zone, zone_size, current_zone, nodes_per_zone, connection)

    next = Int[]
    for i in current_set
        if zone_size[current_zone] >= nodes_per_zone
            break # zone completed
        end
        for nei in connection[i]
            if zone_size[current_zone] >= nodes_per_zone
                break # zone completed
            end
            if zone[nei] == 0 # then: add to zone
                zone[nei] = current_zone
                zone_size[current_zone] += 1
                push!(next, nei)
            end
        end
    end

    return next
end
