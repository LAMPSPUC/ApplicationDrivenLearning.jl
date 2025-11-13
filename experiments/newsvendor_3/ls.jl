import NNlib

function get_ls_solution(X, Y)
    T, I = size(Y)
    f = size(X, 2)
    p = Int(f / I)
    sol = zeros(Float32, I, p+1)
    costs = zeros(I)
    ls_opt_model = JuMP.Model(Gurobi.Optimizer)
    @variable(ls_opt_model, θ[1:p+1])
    set_silent(ls_opt_model)
    for i=1:I
        yhat = X[:, (i-1)*p+1:i*p] * θ[1:p] .+ θ[p+1]
        @objective(ls_opt_model, Min, sum((yhat .- Y[:, i]).^2) / T)
        optimize!(ls_opt_model)
        sol[i, :] = value.(θ)
        costs[i] = objective_value(ls_opt_model)
    end
    println("Final Err = $(round(sum(costs) / I, digits=2))")
    return sol, costs
end

function train_single_nn(nn, X, Y, rule, epochs, time_limit)
    T = size(X, 1)
    stable_iters = 0
    last_err = 0
    train_data = Flux.DataLoader((X', Y'), batchsize=T)
    opt_state = Flux.setup(rule, nn)
    init_time = time()
    for epoch=1:epochs
        Flux.train!(nn, train_data, opt_state) do m, x, y
            sum((m(x) - y).^2) / T
        end

        # compute error
        err = sum((nn(X') - Y').^2) / T
        # println("Epoch $epoch | Err=$(round(err, digits=2))")

        # check convergence
        err_var = abs(err - last_err)
        last_err = err
        if err_var < 1e-4
            stable_iters += 1
        else
            stable_iters = 0
        end
        if stable_iters > 100
            # println("Pre-training converged after $epoch epochs (err=$err).")
            break
        end

        # check time limit
        if time() - init_time > time_limit
            # println("Pre-training stopped after $(round(time() - init_time, digits=2)) seconds.")
            break
        end
    end
    err = sum((nn(X') - Y').^2) / T    
    return err
end

function get_single_nn(p::Int, hidden_layers::Int, hidden_layers_size::Int=64)
    if hidden_layers == 0
        return Flux.Chain(Flux.Dense(p => 1), Flux.relu)
    else
        return Flux.Chain(
            Flux.Dense(p => hidden_layers_size, Flux.relu),
            [Flux.Dense(hidden_layers_size, hidden_layers_size, Flux.relu) for _ in 1:hidden_layers-1]...,
            Flux.Dense(hidden_layers_size => 1, Flux.relu)
        )
    end
end

function get_nns(p::Int, I::Int, hidden_layers::Int, hidden_layers_size::Int=64)
    return [
        get_single_nn(p, hidden_layers, hidden_layers_size) for _ in 1:I
    ]
end