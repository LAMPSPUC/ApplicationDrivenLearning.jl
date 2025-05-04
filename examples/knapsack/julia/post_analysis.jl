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

costs_2.adl_regret = 100 * (costs_2.opt_cost - costs_2.test_cost) ./ costs_2.opt_cost
costs_2.pyepo_regret = 100 * (costs_2.opt_cost - costs_2.pyepo_cost) ./ costs_2.opt_cost

bins = range(0, 50, length=30)
fig = histogram(costs_2.pyepo_regret, bins=bins, label="ADL", alpha=0.5)
histogram!(costs_2.adl_regret, bins=bins, label="PyEPO", alpha=0.5)
savefig(joinpath(RES2_PATH, "adl_vs_pyeppo_regret_histogram.png"))

fig = boxplot(
    [costs_2.adl_regret costs_2.pyepo_regret],
    label=["ADL" "PyEPO"],
    title="ADL vs PyEPO",
    ylabel="Regret (%)",
    xlabel="Test instance",
    legend=:topright,
)
savefig(joinpath(RES2_PATH, "adl_vs_pyeppo_regret_boxplot.png"))
