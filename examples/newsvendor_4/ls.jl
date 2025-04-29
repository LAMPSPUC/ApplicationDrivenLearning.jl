function get_ls_solution(X, Y)
    T, I = size(Y)
    f = size(X, 2)
    ls_opt_model = JuMP.Model(Gurobi.Optimizer)
    @variable(ls_opt_model, θ[1:I, 1:f])
    yhat = X * θ'
    @objective(ls_opt_model, Min, sum((yhat .- Y).^2) / (T*I))
    optimize!(ls_opt_model)
    return value.(θ)
end

function train_nn(nn, X, Y, rule=Flux.Adam(), epochs=300, time_limit=Inf)
    I = size(Y, 2)
    p = size(X, 2)
    train_data = Flux.DataLoader((X', Y'), batchsize=1)
    opt_state = Flux.setup(rule, nn)
    init_time = time()
    for epoch=1:epochs
        Flux.train!(nn, train_data, opt_state) do m, x, y
            mean((m(x) - y).^2)
        end
        if epoch % 10 == 0       
            err = mean((nn(X') - Y').^2)
            err = round(err, digits=2)
            println("Epoch $epoch | Err = $err")
        end

        if time() - init_time > time_limit
            break
        end
    end
end

function get_nn(p::Int, I::Int, hidden_layers::Int, hidden_layers_size::Int=64)
    Random.seed!(0)

    if hidden_layers == 0
        return Flux.Chain(Flux.Dense(p*I => I; bias=false), Flux.relu)
    else
        return Flux.Chain(
            Flux.Dense(p*I => hidden_layers_size, Flux.relu),
            [Flux.Dense(hidden_layers_size, hidden_layers_size, Flux.relu) for _ in 1:hidden_layers-1]...,
            Flux.Dense(hidden_layers_size => I, Flux.relu; bias=false)
        )
    end
end