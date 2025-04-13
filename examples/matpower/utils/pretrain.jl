# predictive model

if N_HIDDEN_LAYERS == 0
    nns = [
        Flux.Chain(Flux.Dense(lags => 1), NNlib.leakyrelu),
        Flux.Chain(Flux.Dense(1 => 20; bias=false), NNlib.leakyrelu),
    ]
else
    nns = [
        Flux.Chain(
            Flux.Dense(lags => HIDDEN_SIZE), NNlib.leakyrelu,
            [Flux.Dense(HIDDEN_SIZE => HIDDEN_SIZE, NNlib.leakyrelu) for _=1:N_HIDDEN_LAYERS-1]...,
            Flux.Dense(HIDDEN_SIZE => 1),
        ),
        Flux.Chain(Flux.Dense(1 => 20; bias=false, ), NNlib.leakyrelu),
    ]
end
input_output_map = [
    Dict(collect(lags*(d-1)+1:lags*(d-1)+lags) => [d] for d=1:pd.n_demand),
    Dict([lags*pd.n_demand+1] => collect(pd.n_demand+1:pd.n_demand+2*pd.n_zones))
]

# fix nns[2] params
Flux.params(nns[2][1])[1] .= Y_train[1,pd.n_demand+1:end]

# pre-train
if PRETRAIN_BATCH_SIZE == -1
    pbs = size(X_train, 1)
else
    pbs = PRETRAIN_BATCH_SIZE
end
if pretrain
    train_data = Flux.DataLoader((X_train', Y_train'), batchsize=pbs, shuffle=true)
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
        
        err2 = mean(sum((nns[1](X_train[:, lags*(d-1)+1:lags*(d-1)+lags]') - Y_train[:, d]').^2 for d=1:pd.n_demand))
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
