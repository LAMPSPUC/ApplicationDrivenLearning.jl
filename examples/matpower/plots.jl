import Plots

# get LS model
models_state = JLD2.load(pretrained_model_state, "state")
Flux.loadmodel!(nns, models_state);
pred_model = ADL.PredictiveModel(
    nns, 
    input_output_map, 
    lags*pd.n_demand+1, 
    pd.n_demand+2*pd.n_zones
)
ADL.set_forecast_model(
    model,
    deepcopy(pred_model)
)
ls_pred = model.forecast(X')'
ls_cost = ADL.compute_cost(model, X, Y)

# get OPT model
models_state = JLD2.load(final_model_state, "state")
Flux.loadmodel!(nns, models_state);
pred_model = ADL.PredictiveModel(
    nns, 
    input_output_map, 
    lags*pd.n_demand+1, 
    pd.n_demand+2*pd.n_zones
)
ADL.set_forecast_model(
    model,
    deepcopy(pred_model)
)
opt_pred = model.forecast(X')'
opt_cost = ADL.compute_cost(model, X, Y)

diff_pred = opt_pred - ls_pred
pdiff_pred = diff_pred ./ ls_pred
fig1 = histogram(
    100 .* vec(pdiff_pred[:, 1:pd.n_demand]),
    alpha=.7,
    title="Percentual difference between OPT and LS predictions",
    xlabel="%",
    label=""
)
savefig(fig1, joinpath(imgs_path, "pred_diff_hist.png"))

up_res_pdiff = filter(
    x -> !isnan(x),
    vec(pdiff_pred[:, pd.n_demand+1:pd.n_demand+pd.n_zones])
)
fig2 = histogram(
    100*up_res_pdiff,
    alpha=.7,
    title="Percentual difference between OPT and LS ResUp",
    xlabel="%",
    label=""
)
savefig(fig2, joinpath(imgs_path, "pred_diff_resup.png"))
dn_res_pdiff = filter(
    x -> !isnan(x),
    vec(pdiff_pred[:, pd.n_demand+pd.n_zones+1:end])
)
fig3 = histogram(
    100*dn_res_pdiff,
    alpha=.7,
    title="Percentual difference between OPT and LS ResDown",
    xlabel="%",
    label=""
)
savefig(fig3, joinpath(imgs_path, "pred_diff_resdn.png"))