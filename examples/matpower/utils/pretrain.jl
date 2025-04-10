# predictive model
weights_dist = Distributions.Exponential(1.0)
weights_init(size...) = rand(weights_dist, size)
nns = [
    Flux.Chain(Flux.Dense(lags => 1; init=weights_init), NNlib.leakyrelu),
    Flux.Chain(Flux.Dense(1 => 20; bias=false, init=weights_init), NNlib.leakyrelu),
]
input_output_map = [
    Dict(collect(lags*(d-1)+1:lags*(d-1)+lags) => [d] for d=1:pd.n_demand),
    Dict([lags*pd.n_demand+1] => collect(pd.n_demand+1:pd.n_demand+2*pd.n_zones))
]

# fix nns[2] params
Flux.params(nns[2][1])[1] .= Y[1,pd.n_demand+1:end]

# pre-train
if PRETRAIN_BATCH_SIZE == -1
    pbs = size(X, 1)
else
    pbs = PRETRAIN_BATCH_SIZE
end
if pretrain
    train_data = Flux.DataLoader((X', Y'), batchsize=pbs, shuffle=true)
    opt_state = Flux.setup(Flux.Adam(PRETRAIN_LEARNING_RATE), nns)
    local epoch = 1
    local init_time = time()
    local err = 1e8
    local err_var = 1e8
    while (epoch <= PRETRAIN_EPOCHS) & (err_var >= 1e-2)
        Flux.train!(nns, train_data, opt_state) do m, x, y
            sum(
                sum((m[1](x[lags*(d-1)+1:lags*(d-1)+lags,:]) - y[d, :]').^2)
                for d=1:pd.n_demand
            )
        end
        
        err2 = mean(sum((nns[1](X[:, lags*(d-1)+1:lags*(d-1)+lags]') - Y[:, d]').^2 for d=1:pd.n_demand))
        err_var = abs(err - err2)
        err = err2

        if epoch % 50 == 0
            println("Epoch $epoch | Err = $(round(err, digits=2))")
        end
        epoch += 1

        if (time() - init_time) > PRETRAIN_MAX_TIME
            println("Pre-training time limit reached. Stopping pre-training.")
            break
        end
    end

    # store models states
    JLD2.jldsave(pretrained_model_state; state=Flux.state(nns))
else
    models_state = JLD2.load(pretrained_model_state, "state")
    Flux.loadmodel!(nns, models_state);
end
