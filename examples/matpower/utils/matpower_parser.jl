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

Simple parser of matpower .m files.
This data is subsequently used by builder_system_data.jl
=#

mutable struct Bus
    bus_i::Int64
    type::Int64
    Pd::Float64
    Qd::Float64
    Gs::Float64
    Bs::Float64
    area::Int64
    Vm::Float64
    Va::Float64
    baseKV::Float64
    zone::Int64
    Vmax::Float64
    Vmin::Float64
    function Bus(Arrayattr)
        bus_i = parse(Int64, Arrayattr[1])
        type = parse(Int64, Arrayattr[2])
        Pd = parse(Float64, Arrayattr[3])
        Qd = parse(Float64, Arrayattr[4])
        Gs = parse(Float64, Arrayattr[5])
        Bs = parse(Float64, Arrayattr[6])
        area = parse(Int64, Arrayattr[7])
        Vm = parse(Float64, Arrayattr[8])
        Va = parse(Float64, Arrayattr[9])
        baseKV = parse(Float64, Arrayattr[10])
        zone = parse(Int64, Arrayattr[11])
        Vmax = parse(Float64, Arrayattr[12])
        Vmin = parse(Float64, Arrayattr[13])
        new(bus_i, type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin)
    end
end

mutable struct Generator
    bus::Int64
    Pg::Float64
    Qg::Float64
    Qmax::Float64
    Qmin::Float64
    Vg::Float64
    mBase::Float64
    status::Int64
    Pmax::Float64
    Pmin::Float64
    function Generator(Arrayattr)
        bus = parse(Int64, Arrayattr[1])
        Pg = parse(Float64, Arrayattr[2])
        Qg = parse(Float64, Arrayattr[3])
        Qmax = parse(Float64, Arrayattr[4])
        Qmin = parse(Float64, Arrayattr[5])
        Vg = parse(Float64, Arrayattr[6])
        mBase = parse(Float64, Arrayattr[7])
        status = parse(Int64, Arrayattr[8])
        Pmax = parse(Float64, Arrayattr[9])
        Pmin = parse(Float64, Arrayattr[10])
        new(bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin)
    end
end

mutable struct GeneratorCost
    two::Int64
    startup::Float64
    shutdown::Float64
    n::Int64
    c2::Float64
    c1::Float64
    c0::Float64
    function GeneratorCost(Arrayattr)
        two = parse(Int64, Arrayattr[1])
        startup = parse(Float64, Arrayattr[2])
        shutdown = parse(Float64, Arrayattr[3])
        n = parse(Int64, Arrayattr[4])
        c2 = parse(Float64, Arrayattr[5])
        c1 = parse(Float64, Arrayattr[6])
        c0 = parse(Float64, Arrayattr[7])
        new(two, startup, shutdown, n, c2, c1, c0)
    end
end

mutable struct Branch
    fbus::Int64
    tbus::Int64
    r::Float64
    x::Float64
    b::Float64
    rateA::Float64
    rateB::Float64
    rateC::Float64
    ratio::Float64
    angle::Float64
    status::Int64
    angmin::Float64
    angmax::Float64
    function Branch(Arrayattr)
        fbus = parse(Int64, Arrayattr[1])
        tbus = parse(Int64, Arrayattr[2])
        r = parse(Float64, Arrayattr[3])
        x = parse(Float64, Arrayattr[4])
        b = parse(Float64, Arrayattr[5])
        rateA = parse(Float64, Arrayattr[6])
        rateB = parse(Float64, Arrayattr[7])
        rateC = parse(Float64, Arrayattr[8])
        ratio = parse(Float64, Arrayattr[9])
        angle = parse(Float64, Arrayattr[10])
        status = parse(Int64, Arrayattr[11])
        angmin = parse(Float64, Arrayattr[12])
        angmax = parse(Float64, Arrayattr[13])
        new(fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax)
    end
end

mutable struct Matpower
    bus::Vector{Bus}
    gen::Vector{Generator}
    gencost::Vector{GeneratorCost}
    branch::Vector{Branch}
end

function read_matlab_file(prob_file::String)
    f = open(prob_file, "r+")
    filetext = readlines(prob_file, keep = true)

    bus_read = false
    gen_read = false
    gencost_read = false
    branch_read = false
    Buses = Bus[]
    Generators = Generator[]
    GeneratorCosts = GeneratorCost[]
    Branches = Branch[]
    for line in filetext
        line_t = replace(line, ";\n" => "")
        line_t = replace(line_t, ";" => "")
        attr = [i for i in split(line_t, " ") if i != ""]
        if occursin("];", line)
            bus_read = false
            gen_read = false
            gencost_read = false
            branch_read = false
        end

        if bus_read
            new_bus = Bus(attr)
            push!(Buses, new_bus)
        end

        if gen_read
            new_gen = Generator(attr)
            push!(Generators, new_gen)
        end

        if gencost_read
            new_gencost = GeneratorCost(attr)
            push!(GeneratorCosts, new_gencost)
        end

        if branch_read
            new_branch = Branch(attr)
            push!(Branches, new_branch)
        end

        if occursin("mpc.bus ", line)
            bus_read = true
        end

        if occursin("mpc.gen ", line)
            gen_read = true
        end

        if occursin("mpc.gencost ", line)
            gencost_read = true
        end

        if occursin("mpc.branch ", line)
            branch_read = true
        end
    end

    return Matpower(Buses, Generators, GeneratorCosts, Branches)
end
