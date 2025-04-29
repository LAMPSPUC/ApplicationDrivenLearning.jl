import Flux
import Gurobi
import Random
import BilevelJuMP
import Distributions
using CSV
using JuMP
using DataFrames
using Statistics
using ApplicationDrivenLearning

include("config.jl")
include("data.jl")
include("lp.jl")
include("ls.jl")

# paths
IMGS_PATH = joinpath(@__DIR__, "imgs")
if !isdir(IMGS_PATH)
    mkdir(IMGS_PATH)
end

rows = []
for I in I_space

    # generate_data
    X_train, Y_train = generate_series_data(I, T_train, r, p)
    X_test, Y_test = generate_series_data(I, T_test, r, p)

    # init model
    model = init_newsvendor_model(I, Gurobi.Optimizer);

    for n_hidden_layers in n_hidden_layers_space
        # start model
        ls_nn = get_nn(p, I, n_hidden_layers, hidden_layers_size)
        layers_size = size.(Flux.params(ls_nn))
        n_params = sum(prod.(layers_size))
    
        # train nn with least-squares
        t0 = time()
        if n_hidden_layers == 0
            ls_θ = get_ls_solution(X_train, Y_train)
            Flux.params(ls_nn)[1] .= ls_θ
        else
            train_nn(ls_nn, X_train, Y_train, Flux.Adam(), max_iter, pretrain_time_limit)
        end
        time_ls = time() - t0
        
        # set forecast model
        input_output_map = Dict(collect(1:size(X_train, 2)) => collect(1:size(Y_train, 2)))
        ApplicationDrivenLearning.set_forecast_model(
            model,
            ApplicationDrivenLearning.PredictiveModel(deepcopy(ls_nn), input_output_map),
            
        )

        # ls prediction
        yhat_ls_train = model.forecast(X_train')'
        yerr_ls_train = Statistics.mean(sum((yhat_ls_train - Y_train).^2, dims=2))
        cost_ls_train = ApplicationDrivenLearning.compute_cost(model, X_train, Y_train)
        yhat_ls_test = model.forecast(X_test')'
        yerr_ls_test = Statistics.mean(sum((yhat_ls_test - Y_test).^2, dims=2))
        cost_ls_test = ApplicationDrivenLearning.compute_cost(model, X_test, Y_test)

        # train with bilevel mode
        if n_hidden_layers == 0
            # set forecast model without relu
            ApplicationDrivenLearning.set_forecast_model(
                model,
                ApplicationDrivenLearning.PredictiveModel(deepcopy(Flux.Chain(ls_nn[1]))),
                
            )
            t0 = time()
            bl_sol = ApplicationDrivenLearning.train!(
                model, X_train, Y_train,
                ApplicationDrivenLearning.Options(
                    ApplicationDrivenLearning.BilevelMode,
                    optimizer=Gurobi.Optimizer,
                    mode=BilevelJuMP.FortunyAmatMcCarlMode(primal_big_M=5000, dual_big_M=5000)
                )
            )
            time_bl = time() - t0
            yhat_bl_train = model.forecast(X_train')'
            yerr_bl_train = Statistics.mean(sum((yhat_bl_train - Y_train).^2, dims=2))
            cost_bl_train = ApplicationDrivenLearning.compute_cost(model, X_train, Y_train)
            yhat_bl_test = model.forecast(X_test')'
            yerr_bl_test = Statistics.mean(sum((yhat_bl_test - Y_test).^2, dims=2))
            cost_bl_test = ApplicationDrivenLearning.compute_cost(model, X_test, Y_test)
        else
            time_bl = NaN
            yerr_bl_train = NaN
            cost_bl_train = NaN
            time_bl_train = NaN
            yerr_bl_test = NaN
            cost_bl_test = NaN
            time_bl_test = NaN
        end

        # train with nelder mead
        if n_params <= max_params_nelder_mead
            ApplicationDrivenLearning.set_forecast_model(
                model,
                ApplicationDrivenLearning.PredictiveModel(
                    deepcopy(ls_nn), 
                    input_output_map
                )
            )
            t0 = time()
            nm_sol = ApplicationDrivenLearning.train!(
                model, X_train, Y_train,
                ApplicationDrivenLearning.Options(
                    ApplicationDrivenLearning.NelderMeadMode,
                    iterations=max_iter, 
                    show_trace=true, 
                    show_every=compute_every,
                    time_limit=time_limit,
                )
            )
            time_nm = time() - t0
            yhat_nm_train = model.forecast(X_train')'
            yerr_nm_train = Statistics.mean(sum((yhat_nm_train - Y_train).^2, dims=2))
            cost_nm_train = ApplicationDrivenLearning.compute_cost(model, X_train, Y_train)
            yhat_nm_test = model.forecast(X_test')'
            yerr_nm_test = Statistics.mean(sum((yhat_nm_test - Y_test).^2, dims=2))
            cost_nm_test = ApplicationDrivenLearning.compute_cost(model, X_test, Y_test)
        else
            time_nm = NaN
            yerr_nm_train = NaN
            cost_nm_train = NaN
            time_nm_train = NaN
            yerr_nm_test = NaN
            cost_nm_test = NaN
            time_nm_test = NaN
        end

        # train with gradient descent
        ApplicationDrivenLearning.set_forecast_model(
            model,
            ApplicationDrivenLearning.PredictiveModel(deepcopy(ls_nn), input_output_map)
        )
        t0 = time()
        gd_sol = ApplicationDrivenLearning.train!(
            model, X_train, Y_train,
            ApplicationDrivenLearning.Options(
                ApplicationDrivenLearning.GradientMode,
                rule=Flux.Adam(lr),
                batch_size=batch_size,
                compute_cost_every=compute_every,
                epochs=max_iter,
                time_limit=time_limit
            )
        )
        time_gd = time() - t0
        yhat_gd_train = model.forecast(X_train')'
        yerr_gd_train = Statistics.mean(sum((yhat_gd_train - Y_train).^2, dims=2))
        cost_gd_train = ApplicationDrivenLearning.compute_cost(model, X_train, Y_train)
        yhat_gd_test = model.forecast(X_test')'
        yerr_gd_test = Statistics.mean(sum((yhat_gd_test - Y_test).^2, dims=2))
        cost_gd_test = ApplicationDrivenLearning.compute_cost(model, X_test, Y_test)

        push!(rows, (
            I,
            n_hidden_layers, 
            n_params, 
            time_ls,yerr_ls_train,cost_ls_train,yerr_ls_test,cost_ls_test,
            time_bl,yerr_bl_train,cost_bl_train,yerr_bl_test,cost_bl_test,
            time_nm,yerr_nm_train,cost_nm_train,yerr_nm_test,cost_nm_test,
            time_gd,yerr_gd_train,cost_gd_train,yerr_gd_test,cost_gd_test
        ))

    end
end


df = DataFrame(
    rows, 
    [
        :I, :n_layers, :n_params,
        :time_ls, :yerr_ls_train, :cost_ls_train, :yerr_ls_test, :cost_ls_test,
        :time_bl, :yerr_bl_train, :cost_bl_train, :yerr_bl_test, :cost_bl_test,
        :time_nm, :yerr_nm_train, :cost_nm_train, :yerr_nm_test, :cost_nm_test,
        :time_gd, :yerr_gd_train, :cost_gd_train, :yerr_gd_test, :cost_gd_test
    ]
)

# save as csv
CSV.write(joinpath(IMGS_PATH, "newsvendor_4.csv"), df)
