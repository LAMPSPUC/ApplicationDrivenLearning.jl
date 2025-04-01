function get_ls_solution(X, Y)
    T, I = size(Y)
    p = Int(size(X, 2) / I)
    ls_opt_model = JuMP.Model(Gurobi.Optimizer)
    @variable(ls_opt_model, θ[1:p+1])
    yhat = Matrix{AffExpr}(undef, T, I)
    for i=1:I
        yhat[:, i] .= X[:, (i-1)*p+1:i*p] * θ[1:end-1] .+ θ[end]
    end
    @objective(ls_opt_model, Min, sum((yhat .- Y).^2) / T)
    optimize!(ls_opt_model)
    return value.(θ)
end

function train_nn(nn, X, Y, rule=Flux.Adam(), epochs=300, time_limit=Inf)
    I = size(Y, 2)
    p = Int(size(X, 2) // I)
    train_data = Flux.DataLoader((X', Y'), batchsize=1)
    opt_state = Flux.setup(rule, nn)
    init_time = time()
    for epoch=1:epochs
        Flux.train!(nn, train_data, opt_state) do m, x, y
            Statistics.mean(
                sum([
                    (m(x[(i-1)*p+1:i*p, :]) - y[i, :]).^2 
                    for i=1:I
                ])
            )
        end
        if epoch % 10 == 0       
            err = Statistics.mean(
                sum([
                    (nn(X[:, (i-1)*p+1:i*p]') .- Y[:, i]').^2
                    for i=1:I
                ])
            )
            err = round(err, digits=2)
            println("Epoch $epoch | Err = $err")
        end

        if time() - init_time > time_limit
            break
        end
    end
end