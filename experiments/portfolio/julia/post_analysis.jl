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

costs_2.pyepo_cost = -costs_1.cost

costs_2.adl_regret = (costs_2.test_cost - costs_2.opt_cost) ./ abs.(costs_2.opt_cost)
costs_2.pyepo_regret = (costs_2.pyepo_cost - costs_2.opt_cost) ./ abs.(costs_2.opt_cost)
costs_2.adl_improv = (costs_2.ls_cost - costs_2.test_cost) ./ abs.(costs_2.ls_cost)
costs_2.pyepo_improv = (costs_2.ls_cost - costs_2.pyepo_cost) ./ abs.(costs_2.ls_cost)

adl_avg_regret = round(mean(costs_2.adl_regret), digits=2)
adl_median_regret = round(median(costs_2.adl_regret), digits=2)
adl_std_regret = round(std(costs_2.adl_regret), digits=2)

adl_avg_improv = round(mean(costs_2.adl_improv), digits=2)
adl_median_improv = round(median(costs_2.adl_improv), digits=2)
adl_std_improv = round(std(costs_2.adl_improv), digits=2)

pyepo_avg_regret = round(mean(costs_2.pyepo_regret), digits=2)
pyepo_median_regret = round(median(costs_2.pyepo_regret), digits=2)
pyepo_std_regret = round(std(costs_2.pyepo_regret), digits=2)

pyepo_avg_improv = round(mean(costs_2.pyepo_improv), digits=2)
pyepo_median_improv = round(median(costs_2.pyepo_improv), digits=2)
pyepo_std_improv = round(std(costs_2.pyepo_improv), digits=2)

adl_wins = sum(costs_2.test_cost .< costs_2.pyepo_cost)
pyepo_wins = sum(costs_2.test_cost .> costs_2.pyepo_cost)

println("ADL Regret: avg=$adl_avg_regret, median=$adl_median_regret, std=$adl_std_regret")
println("PyEPO Regret: avg=$pyepo_avg_regret, median=$pyepo_median_regret, std=$pyepo_std_regret")
println("ADL Improvement: avg=$adl_avg_improv, median=$adl_median_improv, std=$adl_std_improv")
println("PyEPO Improvement: avg=$pyepo_avg_improv, median=$pyepo_median_improv, std=$pyepo_std_improv")
println("ADL Wins: $adl_wins ($(round(100*adl_wins / size(costs_2, 1), digits=2))%)")
println("PyEPO Wins: $pyepo_wins ($(round(100*pyepo_wins / size(costs_2, 1), digits=2))%)")

# costs scatterplot
fig = plot(
    title="Costs Comparison",
    xlabel="PyEPO",
    ylabel="ApplicationDriven",
)
scatter!(costs_2.pyepo_cost, costs_2.test_cost, label="")
savefig(joinpath(RES2_PATH, "costs_scatter.png"))

# plot improvement
min_val = minimum(Array(costs_2[:, [:adl_improv, :pyepo_improv]]))
max_val = maximum(Array(costs_2[:, [:adl_improv, :pyepo_improv]]))
bins = range(min_val*0.99, max_val*1.01, length=30)
fig = plot(
    title="Improvement Comparison",
    xlabel="Improvement",
    ylabel="Frequency",
)
histogram!(costs_2.pyepo_improv, bins=bins, label="PyEPO (avg=$pyepo_avg_improv)", alpha=0.7, color=:grey)
histogram!(costs_2.adl_improv, bins=bins, label="ADL (avg=$adl_avg_improv)", alpha=0.7)
savefig(joinpath(RES2_PATH, "adl_vs_pyeppo_improv_histogram.png"))

fig = boxplot(
    [costs_2.pyepo_improv costs_2.adl_improv],
    label=["PyEPO (avg=$pyepo_avg_improv)" "ADL (avg=$adl_avg_improv)"],
    title="Improvement Comparison",
    ylabel="Improvement",
    xlabel="Package",
    legend=:topleft,
)
savefig(joinpath(RES2_PATH, "adl_vs_pyeppo_improv_boxplot.png"))

# plot normalized regret
max_val = maximum(Array(costs_2[:, [:adl_regret, :pyepo_regret]]))
bins = range(0, max_val*1.01, length=30)
fig = plot(
    title="Normalized Regret Comparison",
    xlabel="Normalized Regret",
    ylabel="Frequency",
)
histogram!(costs_2.pyepo_regret, bins=bins, label="PyEPO (avg=$pyepo_avg_regret)", alpha=0.7, color=:grey)
histogram!(costs_2.adl_regret, bins=bins, label="ADL (avg=$adl_avg_regret)", alpha=0.7)
savefig(joinpath(RES2_PATH, "adl_vs_pyeppo_regret_histogram.png"))

fig = boxplot(
    [costs_2.pyepo_regret costs_2.adl_regret],
    label=["PyEPO (avg=$pyepo_avg_regret)" "ADL (avg=$adl_avg_regret)"],
    title="Normalized Regret Comparison",
    ylabel="Normalized Regret",
    xlabel="Package",
    legend=:topleft,
)
savefig(joinpath(RES2_PATH, "adl_vs_pyeppo_regret_boxplot.png"))
