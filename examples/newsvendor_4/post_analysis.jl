import Random
using CSV
using Plots
using Query
using StatsPlots
using DataFrames
using Statistics

# paths
IMGS_PATH = joinpath(@__DIR__, "results")
if !isdir(IMGS_PATH)
    mkdir(IMGS_PATH)
end

function replace_I_with_count(df)
    ci = 1
    for i in unique(plt_df.I)
        replace!(plt_df.I, i => ci)
        ci+=1
    end
end

# load dataframe from csv
df = CSV.read(joinpath(IMGS_PATH, "newsvendor_4.csv"), DataFrame)

# train cost vs. number of hidden layers
df.improve_bl_train = 100 * (df.cost_bl_train .- df.cost_ls_train) ./ df.cost_ls_train
df.improve_nm_train = 100 * (df.cost_nm_train .- df.cost_ls_train) ./ df.cost_ls_train
df.improve_gd_train = 100 * (df.cost_gd_train .- df.cost_ls_train) ./ df.cost_ls_train
df.improve_bl_test = 100 * (df.cost_bl_test .- df.cost_ls_test) ./ df.cost_ls_test
df.improve_nm_test = 100 * (df.cost_nm_test .- df.cost_ls_test) ./ df.cost_ls_test
df.improve_gd_test = 100 * (df.cost_gd_test .- df.cost_ls_test) ./ df.cost_ls_test

# plot train improvement from LS by model and problem scale
plt_df = stack(
    df[:, [
        :I, 
        :improve_bl_train, :improve_nm_train, :improve_gd_train
    ]], 
    [
        :improve_bl_train, :improve_nm_train, :improve_gd_train
    ],
    variable_name=:model
)
plt_df = filter(:value => !isnan, plt_df)
sort!(plt_df, [:I])
replace_I_with_count(plt_df)

@df plt_df groupedbar(
    :I, :value, group=:model, 
    bar_position=:dodge,
    title="Improvement on train data",
    xlabel="Problem scale (I)",
    ylabel="Improvement (%)",
    legend=:topleft,
    label=["Bilevel" "Nelder-Mead" "Gradient Descent"],
    ylims=(0, maximum(plt_df.value) * 1.3),
)
savefig(joinpath(IMGS_PATH, "train_improvement.png"))

# plot test improvement from LS by model and problem scale
plt_df = stack(
    df[:, [
        :I, 
        :improve_bl_test, :improve_nm_test, :improve_gd_test
    ]], 
    [
        :improve_bl_test, :improve_nm_test, :improve_gd_test
    ],
    variable_name=:model
)
plt_df = filter(:value => !isnan, plt_df)
sort!(plt_df, [:I])
replace_I_with_count(plt_df)

@df plt_df groupedbar(
    :I, :value, group=:model, 
    bar_position=:dodge,
    title="Improvement on test data",
    xlabel="Problem scale (I)",
    ylabel="Improvement (%)",
    legend=:topleft,
    label=["Bilevel" "Nelder-Mead" "Gradient Descent"],
    ylims=(0, maximum(plt_df.value) * 1.3),
)
savefig(joinpath(IMGS_PATH, "test_improvement.png"))
