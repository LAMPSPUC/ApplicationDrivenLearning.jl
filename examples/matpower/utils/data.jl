# structures data from ts.d_train into X and Y matrices
Random.seed!(0)
lags = N_LAGS

function get_matrices_data(series, θru_st, θrd_st)
    T = size(series, 2) - lags
    Y = zeros((T, pd.n_demand+2*pd.n_zones))
    for t=1:T
        for d=1:pd.n_demand
            Y[t, d] = series[d, t]
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
                X[t, lags*(d-1)+l] = series[d, t-l]
            end
        end
    end
    X = Float32.(X)
    Y = Float32.(Y)
    return X, Y
end

X_train, Y_train = get_matrices_data(ts.d_train, θru_st, θrd_st)
X_test, Y_test = get_matrices_data(ts.d_test, θru_st, θrd_st)

# normalize data
mu_X = mean(X_train, dims=1)
mu_X[end] = 0.0

std_X = std(X_train, dims=1)
std_X[std_X .== 0] .= 1.0

mu_Y = mean(Y_train, dims=1)

std_Y = std(Y_train, dims=1)
std_Y[std_Y .== 0] .= 1.0

X_train = (X_train .- mu_X) ./ std_X
X_test = (X_test .- mu_X) ./ std_X

Y_train = (Y_train .- mu_Y) ./ std_Y
Y_test = (Y_test .- mu_Y) ./ std_Y
