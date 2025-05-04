import CSV
using Plots
using StatsPlots
using DataFrames
using Statistics


# paths
DATA_PATH = joinpath(@__DIR__, "..//data")
RES1_PATH = joinpath(DATA_PATH, "pyepo_result")
RES2_PATH = joinpath(DATA_PATH, "adl_result")

# load results from pyepo runs
costs_1 = CSV.read(joinpath(RES1_PATH, "costs.csv"), DataFrame)
costs_2 = CSV.read(joinpath(RES2_PATH, "costs.csv"), DataFrame)

costs_2.pyepo_cost = costs_1.cost

costs_2.adl_regret = (costs_2.opt_cost - costs_2.test_cost) ./ costs_2.opt_cost
costs_2.pyepo_regret = (costs_2.opt_cost - costs_2.pyepo_cost) ./ costs_2.opt_cost

adl_avg_regret = round(mean(costs_2.adl_regret), digits=2)
pyepo_avg_regret = round(mean(costs_2.pyepo_regret), digits=2)

max_val = maximum(Array(costs_2[:, [:adl_regret, :pyepo_regret]]))
bins = range(0, max_val, length=30)
fig = plot(
    title="Normalized Regret Comparison",
    xlabel="Normalized Regret (%)",
    ylabel="Frequency",
)
histogram!(costs_2.pyepo_regret, bins=bins, label="PyEPO (avg=$pyepo_avg_regret)", alpha=0.5)
histogram!(costs_2.adl_regret, bins=bins, label="ADL (avg=$adl_avg_regret)", alpha=0.5)
savefig(joinpath(RES2_PATH, "adl_vs_pyeppo_regret_histogram.png"))

fig = boxplot(
    [costs_2.pyepo_regret costs_2.adl_regret],
    label=["PyEPO (avg=$pyepo_avg_regret)" "ADL (avg=$adl_avg_regret)"],
    title="Normalized Regret Comparison",
    ylabel="Normalized Regret (%)",
    xlabel="Package",
    legend=:topleft,
)
savefig(joinpath(RES2_PATH, "adl_vs_pyeppo_regret_boxplot.png"))
