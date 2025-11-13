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

This file is responsible for creating demand data for case studies.
Two options are available:
1 - synthetic AR(1) data
2 - data read from a csv file
=#

Parameters.@with_kw mutable struct TimeSeriesData
    θd_real::OffsetArray{Float64,2,Array{Float64,2}}
    d_test::OffsetArray{Float64,2,Array{Float64,2}}
    d_train::OffsetArray{Float64,2,Array{Float64,2}}
end

function demand_time_series_synthetic(ps_data, tsg, sample::Integer = 1)

    n_demand_lags = ps_data.n_demand_lags
    if !(0 <= n_demand_lags <= 1)
        error(
            "Demand AR lags for synthetic data model must be 0 or 1, bug got $n_demand_lags.",
        )
    end

    max_lags = max(n_demand_lags, ps_data.n_reserve_lags)

    # fixed random generator
    rng = Random.MersenneTwister(1234)

    μLT = ps_data.demand_value # Long-term average demand
    coef_of_variation = tsg.coef_of_variation

    n_demand = ps_data.n_demand

    stages_test = tsg.stages_test
    stages_train = tsg.stages_train
    buffer = tsg.buffer

    T = stages_test + stages_train + 3 * buffer

    # demand data
    d = OffsetArray{Float64}(undef, 1:n_demand, (1-max_lags):T)
    fill!(d, 0.0)

    θd_real = OffsetArray{Float64}(undef, 1:n_demand, 0:n_demand_lags)
    fill!(θd_real, 0.0)

    d[1:n_demand, 0] = μLT

    sig = coef_of_variation .* μLT
    if n_demand_lags == 1
        θd_real[:, 1] .= tsg.θd_lag
        θd_real[:, 0] .= μLT .* (1 .- θd_real[1, 1])
        σ = sqrt.((sig .^ 2) .* (1 .- θd_real[:, 1] .^ 2))
    else
        error("only demand lag available is 1")
    end

    range_stages_test = (1-max_lags):stages_test
    first_stages_test = 1 - max_lags + buffer
    last_stages_test = first_stages_test + length(range_stages_test) - 1

    range_stages_train = (1-max_lags):stages_train
    first_stages_train = last_stages_test + buffer
    last_stages_train = first_stages_train + length(range_stages_train) - 1

    for t = 1:T
        if t == first_stages_train
            for _ = 1:((length(range_stages_train)+buffer)*n_demand*(sample-1))
                randn(rng)
            end
        end
        for b = 1:n_demand
            err = randn(rng)
            if !iszero(μLT[b])
                d[b, t] = max(
                    θd_real[b, 0] +
                    sum(θd_real[b, p] * d[b, t-p] for p = 1:n_demand_lags) +
                    err * σ[b],
                    0.0,
                )
            end
        end
    end

    bus_range = axes(d, 1)

    d_test =
        OffsetArray(d[:, first_stages_test:last_stages_test], bus_range, range_stages_test)

    d_train = OffsetArray(
        d[:, first_stages_train:last_stages_train],
        bus_range,
        range_stages_train,
    )

    return TimeSeriesData(θd_real = θd_real, d_test = d_test, d_train = d_train)
end

function demand_time_series_file(ps_data, tsg, demand_path, sample::Integer = 1)

    n_demand_lags = ps_data.n_demand_lags

    max_lags = max(n_demand_lags, ps_data.n_reserve_lags)

    μLT = ps_data.demand_value # Long-term average demand

    n_demand = ps_data.n_demand

    stages_test = tsg.stages_test
    stages_train = tsg.stages_train

    T = stages_test + stages_train

    # check if all number are multiples of 24 (24 hours in a day)
    if n_demand_lags != 24
        error(
            "Demand model lags must be 24 in the case of demand data in file, but got $n_demand_lags.",
        )
    end
    if mod(stages_test, 24) != 0
        error(
            "In the case of demand from data file, the number of stages in the test phase must be a multiple of 24, but got $stages_test",
        )
    end
    if mod(stages_train, 24) != 0
        error(
            "In the case of demand from data file, the number of stages in the training phase must be a multiple of 24, but got $stages_train",
        )
    end

    θd_real = OffsetArray{Float64}(undef, 1:n_demand, 0:n_demand_lags)
    fill!(θd_real, 0.0)

    range_stages_train = (1-max_lags):stages_train
    range_stages_test = (1-max_lags):stages_test

    shift = (sample - 1) * (length(range_stages_train) + length(range_stages_test))

    @assert mod(shift, 24) == 0

    first_stages_train = 1 + shift # only shift here, since other are linked
    last_stages_train = first_stages_train + length(range_stages_train) - 1

    first_stages_test = last_stages_train + 1
    last_stages_test = first_stages_test + length(range_stages_test) - 1

    # read demand from file
    vals = DelimitedFiles.readdlm(demand_path, ',', skipstart = 1)
    if size(vals, 2) < n_demand + 2
        error(
            "Minimum required number of demand time series is $n_demand, but time serie files only contains $(size(vals, 2)-2) time series.",
        )
    end
    d = permutedims(vals[:, 3:(n_demand+2)], (2, 1))

    for b = 1:n_demand
        d[b, :] .*= μLT[b]
    end

    bus_range = axes(d, 1)

    d_test =
        OffsetArray(d[:, first_stages_test:last_stages_test], bus_range, range_stages_test)

    d_train = OffsetArray(
        d[:, first_stages_train:last_stages_train],
        bus_range,
        range_stages_train,
    )

    return TimeSeriesData(θd_real = θd_real, d_test = d_test, d_train = d_train)
end
