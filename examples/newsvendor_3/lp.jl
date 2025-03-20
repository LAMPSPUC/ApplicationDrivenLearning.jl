function init_newsvendor_model(I, Optimizer, allow_trade::Bool=false)
    # get problem data
    c, q, r = generate_problem_data(I)
    println("c=$c")
    println("q=$q")
    println("r=$r")
    # mount ADL model
    model = ApplicationDrivenLearning.Model()
    @variables(model, begin
        x[1:I] ≥ 0, ApplicationDrivenLearning.Policy
        d[1:I] ≥ 0, ApplicationDrivenLearning.Forecast
    end)
    function build_jump_model(jump_model, x, d)
        @variables(jump_model, begin
            y[1:I] ≥ 0
            w[1:I] ≥ 0
        end)
        if allow_trade
            @variables(jump_model, begin
                z[1:I] ≥ 0
                f[1:I] ≥ 0
                F[1:I, 1:I]
            end)
        end
        @constraints(jump_model, begin
            con1[i=1:I], y[i] ≤ d[i]
            con2[i=1:I], y[i] + w[i] ≤ x[i]
        end)
        if allow_trade
            @constraints(jump_model, begin
            con3[i=1:I], z[i] == x[i] - sum(F[i, :]) + sum(F[:, i])
            con4[i=1:I], f[i] ≥ sum(F[i, :]) 
            end)
        end
        if allow_trade
            cost_exp = @expression(jump_model, c'x-q'y-r'w+c'f)
        else
            cost_exp = @expression(jump_model, c'x-q'y-r'w)
        end
        @objective(jump_model, Min, cost_exp)
    end
    build_jump_model(ApplicationDrivenLearning.Plan(model), [i.plan for i in x], [i.plan for i in d])
    build_jump_model(ApplicationDrivenLearning.Assess(model), [i.assess for i in x], [i.assess for i in d])
    set_optimizer(model, Optimizer)
    set_silent(model)

    return model
end