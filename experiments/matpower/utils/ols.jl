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

Helper functions for least squares (LS) estimation.
The regressions consider multiple lags but they
are univariate, i.e., parameters are estimated
as time series were fully independent.
=#

function regressor_matrix(demand, n_demand_lags, i)
    T = lastindex(demand, 2)
    X = ones(T, 1)
    for p = 1:n_demand_lags
        X = hcat(X, demand[i, (1-p):(T-p)])
    end
    return X
end

function ls(demand, n_demand_lags, i)
    T = lastindex(demand, 2)
    if sum(demand[i, :]) != 0.0
        X = regressor_matrix(demand, n_demand_lags, i)
        θd_ls = OffsetArray(((X'X) \ X' * demand[i, 1:T]), 0:n_demand_lags)
        return θd_ls
    else
        return OffsetArray(zeros(n_demand_lags + 1), 0:n_demand_lags)
    end
end

function multi_ls(demand, n_demand_lags, n_demand)
    t = zeros(0, 1 + n_demand_lags)
    for i = 1:n_demand
        t = vcat(t, ls(demand, n_demand_lags, i).parent')
    end
    OffsetArray(t, 1:n_demand, 0:n_demand_lags)
end

function ls_noise_var(demand, n_demand_lags, n_demand)
    T = lastindex(demand, 2)
    err = zeros(T, 0)
    for i = 1:n_demand
        X = regressor_matrix(demand, n_demand_lags, i)
        θ = ls(demand, n_demand_lags, i)
        err = hcat(err, (demand[i, 1:T] - X * θ.parent))
    end
    err = sum(err, dims = 2)
    return Statistics.std(err)
end

function zone_frac(pd)
    tot = sum(pd.G)
    frac = zeros(pd.n_zones)
    for i = 1:pd.n_generators
        frac[pd.bus_to_zone[pd.generator_to_bus[i]]] += pd.G[i] / tot
    end
    return frac
end

