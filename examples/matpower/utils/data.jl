# structures data from ts.d_train into X and Y matrices
Random.seed!(0)
lags = N_LAGS
T = size(ts.d_train, 2) - 24
Y = zeros((T, pd.n_demand+2*pd.n_zones))
for t=1:T
    for d=1:pd.n_demand
        Y[t, d] = ts.d_train[d, t]
    end
    for z=1:pd.n_zones
        Y[t, pd.n_demand+z] = θru_st[z, 0]
        Y[t, pd.n_demand+z+pd.n_zones] = θrd_st[z, 0]
    end
end
X = ones((T, pd.n_demand*lags+1))
for t=1:T
    for d=1:pd.n_demand
        for l=1:lags
            X[t, lags*(d-1)+l] = ts.d_train[d, t-l]
        end
    end
end
X = Float32.(X)
Y = Float32.(Y)