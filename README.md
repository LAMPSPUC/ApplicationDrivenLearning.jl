# Experiments

This repository contains experiments using ApplicationDrivenLearning.jl.

## Structure


```sh
root/
├─ examples/
│   ├─ newsvendor_1/
│   │   ├─ imgs/ # plots from experiment
│   │   └─ newsvendor_1.jl # main script
│   ├─ newsvendor_2/
│   │   ├─ imgs/ # plots from experiment
│   │   └─ newsvendor_2.jl # main script
│   ├─ newsvendor_3/
│   │   ├─ results/ # data and plots results from experiment
│   │   ├─ config.jl # experiment general parameters
│   │   ├─ data.jl # data generation funcions
│   │   ├─ lp.jl # optimization model mount
│   │   ├─ ls.jl # least-squares model functions
│   │   ├─ post_analysis.jl # plot generation script after computing results
│   │   └─ newsvendor_3.jl # main script
│   ├─ shortest_path/
│   │   ├─ data/ # data folder
│   │   │   ├─ input/ # (x,y) input data
│   │   │   ├─ pyepo_result/ # results from PyEPO
│   │   │   └─ adl_resul/ # results from ApplicationDrivenLearning
│   │   ├─ python/ # python scripts for data generation and PyEPO execution
│   │   ├─ julia/
|   │   │   ├─ shortest_path.jl # ApplicationDrivenLearning execution
|   │   │   └─ post_analysis.jl # final plots generation
│   ├─ knapsack/
│   │   ├─ data/ # data folder
│   │   │   ├─ input/ # (x,y) input data
│   │   │   ├─ pyepo_result/ # results from PyEPO
│   │   │   └─ adl_resul/ # results from ApplicationDrivenLearning
│   │   ├─ python/ # python scripts for data generation and PyEPO execution
│   │   ├─ julia/
|   │   │   ├─ shortest_path.jl # ApplicationDrivenLearning execution
|   │   │   └─ post_analysis.jl # final plots generation
│   ├─ matpower/
│   │   ├─ data/ # data folder
│   │   │   ├─ cases/ # .m files from pglib-opf cases
│   │   │   ├─ results/ # experiment results
│   │   │   └─ demand.csv # historical data for demand series
│   │   ├─ utils/
│   │   ├─ config.jl # experiment parameters
│   │   ├─ main.jl # script for running a full case
│   │   └─ auto_run.jl # script for iteratively running multiple cases
``` 

## Experiments Description

### Newsvendor 1

Simple multistep newsvendor problem with AR-1 process timeseries. Applies least-squares methodology with BilevelMode and shows difference between ls and opt on in-sample prediction, prediction error and assessed cost.

### Nesvendor 2

Uses same basic nesvendor problem, but with 2 timeseries representing 2 different newsvendor instances, with different cost parameters and AR-3 processes for timeseries generation. This shows how to use `input_output_map` to apply the same predictive model for multiple prediction decision variables. 

### Nesvendor 3

Applies multistep newsvendor on multiple problem scales. In this experiment, we compare performance for increasing number of predictive model parameters, with results indicating that GradientMode eventually becomes a better alternative than NelderMeadMode and BilevelMode for big problems.

### Shortest path

Shortest path problem from PyEPO (https://arxiv.org/abs/2206.14234). This takes data constructed from PyEPO code, mounts the same problem, solves it and compare out-of-sample costs.

### Knapsack

Knapsack problem from PyEPO (https://arxiv.org/abs/2206.14234). This takes data constructed from PyEPO code, mounts the same problem, solves it and compare out-of-sample costs. 

### Matpower

Loads system files from PGLib-OPF and train a load forecasting and reserve sizing model in the context of minimal operation cost optimization as described in https://arxiv.org/pdf/2102.13273.