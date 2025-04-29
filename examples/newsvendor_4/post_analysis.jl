import Random
using CSV
using Plots
using StatsPlots
using DataFrames
using Statistics

# paths
IMGS_PATH = joinpath(@__DIR__, "imgs")
if !isdir(IMGS_PATH)
    mkdir(IMGS_PATH)
end

# load dataframe from csv
df = CSV.read(joinpath(IMGS_PATH, "newsvendor_4.csv"), DataFrame)

# train cost vs. number of hidden layers
df.LS = df.cost_ls_train ./ df.I
df.NM = df.cost_nm_train ./ df.I
df.GD = df.cost_gd_train ./ df.I
pvt_df = stack(
    df[:, [:I, :n_layers, :LS, :NM, :GD]], 
    [:LS, :NM, :GD],
    variable_name=:model
)

fig = plot(
    title="Train Cost vs. Number of Hidden Layers", 
    xlabel="Number of Hidden Layers", 
    ylabel="Train Cost", 
    legend=:topright,
    xticks=unique(pvt_df.n_layers),
)
groupedboxplot!(
    pvt_df.n_layers,
    pvt_df.value,
    group=pvt_df.model,
)
savefig(fig, joinpath(IMGS_PATH, "train_costs.png"))

# test cost vs. number of hidden layers
df.LS = df.cost_ls_test ./ df.I
df.NM = df.cost_nm_test ./ df.I
df.GD = df.cost_gd_test ./ df.I
pvt_df = stack(
    df[:, [:I, :n_layers, :LS, :NM, :GD]], 
    [:LS, :NM, :GD],
    variable_name=:model
)

fig = plot(
    title="Test Cost vs. Number of Hidden Layers", 
    xlabel="Number of Hidden Layers", 
    ylabel="Test Cost", 
    legend=:topright,
    xticks=unique(pvt_df.n_layers),
)
groupedboxplot!(
    pvt_df.n_layers,
    pvt_df.value,
    group=pvt_df.model,
)
savefig(fig, joinpath(IMGS_PATH, "test_costs.png"))
