# predictive model

nns = []
if N_HIDDEN_LAYERS == 0
    for i=1:pd.n_demand
        reg = Flux.Chain(Flux.Dense(lags => 1))
        if pretrain
            m = JuMP.Model(Gurobi.Optimizer)
            a = @variable(m, [i=1:lags])
            b = @variable(m)
            err = X_train[:, lags*(i-1)+1:lags*i]*a .+b - Y_train[:, i]
            @objective(m, Min, sum(err.^2))
            set_silent(m)
            optimize!(m)
            Flux.params(reg)[1] .= JuMP.value.(a')
            Flux.params(reg)[2] .= JuMP.value(b)
        end
        push!(nns, reg)
    end
else
    for i=1:pd.n_demand
        push!(nns, Flux.Chain(
            Flux.Dense(lags => HIDDEN_SIZE), Flux.relu,
            [Flux.Dense(HIDDEN_SIZE => HIDDEN_SIZE, Flux.relu) for _=1:N_HIDDEN_LAYERS-1]...,
            Flux.Dense(HIDDEN_SIZE => 1),
        ))
    end
end
push!(nns, Flux.Chain(Flux.Dense(1 => 2*pd.n_zones; bias=false, ),))
input_output_map = Vector{Dict{Vector{Int64}, Vector{Int64}}}()
for i=1:pd.n_demand
    push!(input_output_map, Dict(collect(lags*(i-1)+1:lags*i) => [i]))
end
push!(input_output_map, Dict([lags*pd.n_demand+1] => collect(pd.n_demand+1:pd.n_demand+2*pd.n_zones)))

# fix nns[2] params
Flux.params(nns[end][1])[1] .= Y_train[1,pd.n_demand+1:end]

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
    local stable_iters = 0
    local solution = Flux.params(nns)
    while (epoch <= PRETRAIN_EPOCHS) && (N_HIDDEN_LAYERS > 0)
        Flux.train!(nns, train_data, opt_state) do m, x, y
            sum(
                sum([
                    (m[d](x[lags*(d-1)+1:lags*d, :]) .- y[d, :])^2
                    for d=1:pd.n_demand
                ])
            )
        end
        
        # compute error
        err2 = sum(
            mean(
                (nns[d](X_train[:, lags*(d-1)+1:lags*d]') - Y_train[:, d]').^2
            )
                for d=1:pd.n_demand
        )

        # check convergence
        err_var = abs(err - err2)
        err = err2

        if err_var < 1e-8
            stable_iters += 1
        else
            stable_iters = 0
        end
        if stable_iters > 10
            println("Pre-training converged after $epoch epochs.")
            break
        end

        if epoch % 1 == 0
            println("Epoch $epoch | Err = $(round(err, digits=4))")
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
