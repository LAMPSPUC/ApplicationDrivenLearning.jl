import Random
using CSV
using Plots
using Query
using Glob
using StatsPlots
using DataFrames
using Statistics

# paths
IMGS_PATH = joinpath(@__DIR__, "results")
if !isdir(IMGS_PATH)
    mkdir(IMGS_PATH)
end

# load dataframe from csv files
result_files = filter(file -> occursin("run_", file), readdir(IMGS_PATH))
dfs = []
for f in result_files
    fdf = CSV.read(joinpath(IMGS_PATH, f), DataFrame)
    fdf._id .= rand(1:100000000)
    push!(dfs, fdf)
end
df = vcat(dfs...)

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
        :I, :n_layers,
        :improve_bl_train, :improve_nm_train, :improve_gd_train
    ]], 
    [
        :improve_bl_train, :improve_nm_train, :improve_gd_train
    ],
    variable_name=:model
)
plt_df = filter(:value => !isnan, plt_df)
sort!(plt_df, [:I])
transform!(plt_df, :I => ByRow(string) => :I)
plt_df.model_name = [Dict(
    "improve_bl_train" => "Bilevel",
    "improve_nm_train" => "Nelder-Mead",
    "improve_gd_train" => "Gradient",
)[x] for x in plt_df.model]

@df plt_df groupedboxplot(
    :I, :value, group=:model_name, 
    bar_position=:dodge,
    title="Improvement on train data",
    xlabel="Problem scale (# items)",
    ylabel="Improvement (%)",
    legend=:topright,
    # ylims=(0, maximum(plt_df.value) * 1.3),
)
savefig(joinpath(IMGS_PATH, "train_improvement.png"))

# plot test improvement from LS by model and problem scale
plt_df = stack(
    df[:, [
        :I, :n_layers,
        :improve_bl_test, :improve_nm_test, :improve_gd_test
    ]], 
    [
        :improve_bl_test, :improve_nm_test, :improve_gd_test
    ],
    variable_name=:model
)
plt_df = filter(:value => !isnan, plt_df)
sort!(plt_df, [:I])
transform!(plt_df, :I => ByRow(string) => :I)
plt_df.model_name = [Dict(
    "improve_bl_test" => "Bilevel",
    "improve_nm_test" => "Nelder-Mead",
    "improve_gd_test" => "Gradient",
)[x] for x in plt_df.model]

@df plt_df groupedboxplot(
    :I, :value, group=:model_name, 
    bar_position=:dodge,
    title="Improvement on test data",
    xlabel="Problem scale (# items)",
    ylabel="Improvement (%)",
    legend=:topright,
    # ylims=(0, maximum(plt_df.value) * 1.3),
)
savefig(joinpath(IMGS_PATH, "test_improvement.png"))
