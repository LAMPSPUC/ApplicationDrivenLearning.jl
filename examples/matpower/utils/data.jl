# structures data from ts.d_train into X and Y matrices
Random.seed!(0)
lags = N_LAGS
T = size(ts.d_train, 2) - 24
Y_train = zeros((T, pd.n_demand+2*pd.n_zones))
for t=1:T
    for d=1:pd.n_demand
        Y_train[t, d] = ts.d_train[d, t]
    end
    for z=1:pd.n_zones
        Y_train[t, pd.n_demand+z] = θru_st[z, 0]
        Y_train[t, pd.n_demand+z+pd.n_zones] = θrd_st[z, 0]
    end
end
X_train = ones((T, pd.n_demand*lags+1))
for t=1:T
    for d=1:pd.n_demand
        for l=1:lags
            X_train[t, lags*(d-1)+l] = ts.d_train[d, t-l]
        end
    end
end
X_train = Float32.(X_train)
Y_train = Float32.(Y_train)

# test data
T = size(ts.d_test, 2) - 24
Y_test = zeros((T, pd.n_demand+2*pd.n_zones))
for t=1:T
    for d=1:pd.n_demand
        Y_test[t, d] = ts.d_test[d, t]
    end
    for z=1:pd.n_zones
        Y_test[t, pd.n_demand+z] = θru_st[z, 0]
        Y_test[t, pd.n_demand+z+pd.n_zones] = θrd_st[z, 0]
    end
end
X_test = ones((T, pd.n_demand*lags+1))
for t=1:T
    for d=1:pd.n_demand
        for l=1:lags
            X_test[t, lags*(d-1)+l] = ts.d_test[d, t-l]
        end
    end
end
X_test = Float32.(X_test)
Y_test = Float32.(Y_test)

