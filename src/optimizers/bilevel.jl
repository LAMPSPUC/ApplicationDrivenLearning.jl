using JuMP
using Flux
using BilevelJuMP


function solve_bilevel(
    model::Model,
    X::Matrix{<:Real},
    Y::Matrix{<:Real},
    params::Dict{Symbol, Any}
)

    # extract params
    optimizer = get(params, :optimizer, nothing)
    silent = get(params, :silent, false)
    mode = get(params, :mode, nothing)

    # create bilevel model
    bilevel_model = BilevelJuMP.BilevelModel(
        optimizer,
        mode=mode
    )

    # silence jump model
    if silent
        set_silent(bilevel_model)
    end

    # parameters
    T = size(Y, 1)

    # lower model variables
    low_var_map = Dict{JuMP.VariableRef, Vector{BilevelJuMP.BilevelVariableRef}}()
    for pre_var in all_variables(model.plan)
        low_var_name = string(name(pre_var), "_low")
        low_var_ref = @variable(Lower(bilevel_model), [1:T], base_name=low_var_name)
        if has_lower_bound(pre_var)
            set_lower_bound.(low_var_ref, lower_bound(pre_var))
        end
        if has_upper_bound(pre_var)
            set_upper_bound.(low_var_ref, upper_bound(pre_var))
        end
        low_var_map[pre_var] = low_var_ref
    end

    # upper model variables
    up_var_map = Dict{JuMP.VariableRef, Vector{BilevelJuMP.BilevelVariableRef}}()
    for post_var in all_variables(model.assess)
        if !(post_var in assess_policy_vars(model))
            up_var_name = string(name(post_var), "_up")
            up_var_ref = @variable(Upper(bilevel_model), [1:T], base_name=up_var_name)
            if has_lower_bound(post_var)
                set_lower_bound.(up_var_ref, lower_bound(post_var))
            end
            if has_upper_bound(post_var)
                set_upper_bound.(up_var_ref, upper_bound(post_var))
            end
            up_var_map[post_var] = up_var_ref
        end
    end

    # point to lower model decision variables on upper model
    i_dec_var = 1
    for pre_dec_var in plan_policy_vars(model)
        post_dec_var = assess_policy_vars(model)[i_dec_var]
        up_var_map[post_dec_var] = low_var_map[pre_dec_var]
        i_dec_var += 1
    end

    # lower model base constraints
    for pre_con in JuMP.all_constraints(model.plan, include_variable_in_set_constraints=false)
        pre_con_func = JuMP.constraint_object(pre_con).func
        lhs = [
            value(x -> low_var_map[x][t], pre_con_func)
            for t=1:T
        ]
        @constraint(Lower(bilevel_model), lhs .∈ JuMP.constraint_object(pre_con).set)
    end

    # upper model base constraints
    for post_con in JuMP.all_constraints(model.assess, include_variable_in_set_constraints=false)
        if name(post_con) != "assess_policy_fix"
            post_con_func = JuMP.constraint_object(post_con).func
            lhs = [
                value(x -> up_var_map[x][t], post_con_func)
                for t=1:T
            ]
            @constraint(Upper(bilevel_model), lhs .∈ JuMP.constraint_object(post_con).set)
        end
    end

    # lower model objective
    pre_obj_func = JuMP.objective_function(model.plan)
    pre_obj_sense = JuMP.objective_sense(model.plan)
    low_obj = sum([
        value(x -> low_var_map[x][t], pre_obj_func)
        for t=1:T
    ])
    @objective(Lower(bilevel_model), pre_obj_sense, low_obj)

    # upper model objective
    post_obj_func = JuMP.objective_function(model.assess)
    post_obj_sense = JuMP.objective_sense(model.assess)
    up_obj = sum([
        value(x -> up_var_map[x][t], post_obj_func)
        for t=1:T
    ])
    @objective(Upper(bilevel_model), post_obj_sense, up_obj)

    # fix upper model observations
    i_obs_var = 1
    for obs_var in assess_forecast_vars(model)
        @constraint(Upper(bilevel_model), up_var_map[obs_var] - Y[1:T, i_obs_var] .== 0)
        i_obs_var += 1
    end

    # implement predictive model expression iterating through 
    # layers and creating predictive expression
    layers_inpt = Dict{Any, Any}(
        output_idx => X[1:T, input_idx]
        for (input_idx, output_idx) in model.forecast.input_output_map[1]
    )
    predictive_model_vars = Dict{Int, Any}()
    i_layer = 1
    for layer in model.forecast.networks[1]
        # if it is layer with parameters, process output
        if has_params(layer)
            # get size and parameters W and b
            (layer_size_out, layer_size_in) = size(layer.weight)
            W = @variable(Upper(bilevel_model), [1:layer_size_out, 1:layer_size_in])        
            if layer.bias == false
                b = zeros(layer_size_out)
            else
                b = @variable(Upper(bilevel_model), [1:layer_size_out])
            end
            predictive_model_vars[i_layer] = Dict(:W=>W, :b=>b)
            # build layer output as next layer input
            for output_idx in values(model.forecast.input_output_map[1])
                layers_inpt[output_idx] = layer.σ(W * layers_inpt[output_idx]' .+ b)'
            end
        # if activation function layer, just apply
        elseif supertype(typeof(layer)) == Function
            for output_idx in values(model.forecast.input_output_map[1])
                layers_inpt[output_idx] = layer(layers_inpt[output_idx])
            end
        else
            println("Network layer $ilayer type not supported")
        end
        i_layer += 1
    end
    y_hat = Matrix{Any}(undef, size(Y, 1), size(Y, 2))
    for (output_idx, prediction) in layers_inpt
        y_hat[:, output_idx] = prediction
    end

    # and apply prediction on lower model as constraint
    ipred_var_count = 1
    for pred_var in plan_forecast_vars(model)
        low_pred_var = low_var_map[pred_var] 
        @constraint(Lower(bilevel_model), low_pred_var .- y_hat[:, ipred_var_count] .== 0)
        ipred_var_count += 1
    end

    # solve model
    optimize!(bilevel_model)

    # fix parameters to predictive_model
    ilayer = 1
    for layer in model.forecast.networks[1]
        if has_params(layer)
            for p in Flux.params(layer.weight)
                p .= value.(predictive_model_vars[ilayer][:W])
            end
            for p in Flux.params(layer.bias)
                p .= value.(predictive_model_vars[ilayer][:b])
            end
        end
        ilayer += 1
    end

    return Solution(
        objective_value(bilevel_model), 
        extract_params(model.forecast)
    )
end