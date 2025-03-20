# Examples

This repository contains examples for using ApplicationDrivenLearning.jl.

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
│   │   ├─ imgs/ # plots from experiment
│   │   ├─ data.jl # data generation funcions
│   │   ├─ lp.jl # optimization model mount
│   │   ├─ ls.jl # least-squares model functions
│   │   └─ newsvendor_3.jl # main script
``` 

## Examples Description

### Newvendor 1

Simple multistep newsvendor problem with AR-1 process timeseries. Applies least-squares methodology and BilevelMode and shows difference between ls and opt in in-sample prediction, prediction error and assessed cost. 

### Nesvendor 2

Uses same basic nesvendor problem, but with 2 timeseries representing 2 different newsvendor instances, with different cost parameters and AR-3 processes for timeseries generation. This shows how to use `input_output_map` to apply the same predictive model for multiple prediction decision variables. 

We also analyze the relationship between size of the bias introduced by the application driven learning model, measured by the absolute difference between predictions, and uncertainty from the least-squares model, measured using 95% confidence intervals.

### Nesvendor 3

Uses same problem from `Newsvendor 2` with longer timeseries. In this setting, we compare performance for increasing number of predictive model parameters, showing that GradientMode eventually becomes a better alternative than NelderMeadMode and BilevelMode for big models.
