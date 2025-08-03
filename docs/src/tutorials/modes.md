# Training modes

Training a model with ApplicationDrivenLearning is done by calling the `train!` function. This function has a `mode` argument that represents the algorithm used to train the model. In this tutorial, we will show how to use the different modes and explore the trade-off between them.

The data used in this tutorial is the same as in the [getting started](getting_started.md) tutorial so all the code shown in this page assumes that the full model (`ApplicationDrivenLearning.Model`) is already defined.

## Bilevel mode

The bilevel mode mounts the bilevel optimization problem relative to training the predictive model and uses the `BilevelJuMP` package to solve it. For using this mode, we have to install the `BilevelJuMP` package, import it and set the BilevelJuMP mode (from [BilevelJuMP.jl docs](https://joaquimg.github.io/BilevelJuMP.jl/stable/tutorials/modes/)).

### Arguments

- `optimizer`: JuMP.jl optimizer used to solve the bilevel model using BilevelJuMP.jl.
- `mode`: The mode to use for the bilevel optimization problem. It can be any of the modes supported by the `BilevelJuMP` package.
- `silent`: Whether to print the progress of the bilevel optimization problem.

### Example

```julia
julia> import BilevelJuMP
julia> import HiGHS

julia> opt = ApplicationDrivenLearning.Options(
    ApplicationDrivenLearning.BilevelMode,
    optimizer = HiGHS.Optimizer,
    mode = BilevelJuMP.FortunyAmatMcCarlMode(
        primal_big_M = 100,
        dual_big_M = 100,
    )
);
julia> sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
Running HiGHS 1.9.0 (git hash: 66f735e60): Copyright (c) 2024 HiGHS under MIT licence terms
Coefficient ranges:
  Matrix [1e+00, 2e+04]
  Cost   [5e-01, 5e+00]
  Bound  [1e+00, 1e+00]
  RHS    [5e-01, 1e+02]
Assessing feasibility of MIP using primal feasibility and integrality tolerance of       1e-06
WARNING: Row      0 has         infeasibility of          10 from [lower, value, upper] = 
[             10;               0;              10]
Solution has               num          max          sum
Col     infeasibilities      0            0            0
Integer infeasibilities      0            0            0
Row     infeasibilities      8           20           42
Row     residuals            0            0            0
Attempting to find feasible solution by solving LP for user-supplied values of discrete variables
Coefficient ranges:
  Matrix [1e+00, 2e+04]
  Cost   [5e-01, 5e+00]
  Bound  [0e+00, 0e+00]
  RHS    [5e-01, 1e+02]
Presolving model
12 rows, 7 cols, 24 nonzeros  0s
6 rows, 7 cols, 12 nonzeros  0s
4 rows, 5 cols, 8 nonzeros  0s
4 rows, 5 cols, 8 nonzeros  0s
Presolve : Reductions: rows 4(-32); columns 5(-28); elements 8(-80)
Solving the presolved LP
Using EKK dual simplex solver - serial
  Iteration        Objective     Infeasibilities num(sum)
          0    -3.1250000000e+02 Ph1: 4(5623); Du: 1(0.3125) 0s
          3     3.0000000000e+02 Pr: 0(0) 0s
Solving the original LP from the solution after postsolve
Model status        : Optimal
Simplex   iterations: 3
Objective value     :  3.0000000000e+02
Relative P-D gap    :  0.0000000000e+00
HiGHS run time      :          0.05
Presolving model
24 rows, 17 cols, 52 nonzeros  0s
17 rows, 14 cols, 40 nonzeros  0s
10 rows, 11 cols, 20 nonzeros  0s
4 rows, 5 cols, 8 nonzeros  0s

MIP start solution is feasible, objective value is 300

Solving MIP model with:
   4 rows
   5 cols (0 binary, 0 integer, 0 implied int., 5 continuous)
   8 nonzeros

Src: B => Branching; C => Central rounding; F => Feasibility pump; H => Heuristic; L => Sub-MIP;
     P => Empty MIP; R => Randomized rounding; S => Solve LP; T => Evaluate node; U => Unbounded;
     z => Trivial zero; l => Trivial lower; u => Trivial upper; p => Trivial point        

        Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work
Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts  
 InLp Confl. | LpIters     Time

         0       0         0   0.00%   -inf            300                Large        0  
    0      0         0     0.0s
         1       0         1 100.00%   300             300                0.00%        0  
    0      0         4     0.1s

Solving report
  Status            Optimal
  Primal bound      300
  Dual bound        300
  Gap               0% (tolerance: 0.01%)
  P-D integral      0
  Solution status   feasible
                    300 (objective)
                    0 (bound viol.)
                    0 (int. viol.)
                    0 (row viol.)
  Timing            0.11 (total)
                    0.00 (presolve)
                    0.00 (solve)
                    0.00 (postsolve)
  Max sub-MIP depth 0
  Nodes             1
  Repair LPs        0 (0 feasible; 0 iterations)
  LP iterations     4 (total)
                    0 (strong br.)
                    0 (separation)
                    0 (heuristics)
ApplicationDrivenLearning.Solution(300.0, Real[20.0f0])
```

The output shows the solution found by the optimizer. The first value is the cost of the solution and the second value is the predictive model parameters.

### Pros
- Is the most accurate option as it represents the bilevel optimization problem exactly.

### Cons
- Requires solving a MIP problem that grows in size very fast with the number of data points.
- Relies on parameters specific to the BilevelJuMP package.
- Does not support integer variables in the plan and assess models.

## Nelder-Mead mode

Uses the Nelder-Mead algorithm implemented in the `Optim.jl` package. This algorithm is a gradient-free optimization algorithm that does not require the gradient of the objective function and is very robust to the choice of the initial guess.

### Arguments

- `initial_simplex`: A n + 1 dimensional vector of n-dimensional vectors, where n is
the dimension of predictive model parameters. It will be used to start the search algorithm.
- `parameters`: Used to generate parameters for the algorithm. Same parameter from
Optim implementation of Nelder-Mead.
- Any other parameter acceptable on `Optim.Options` such as `iterations`, `time_limit`
and `g_tol` can be directly passed.

### Example

```julia
julia> opt = ApplicationDrivenLearning.Options(
    ApplicationDrivenLearning.NelderMeadMode,
    show_trace = true
);
julia> sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
Iter     Function value    √(Σ(yᵢ-ȳ)²)/n
------   --------------    --------------
     0     1.473289e+03     5.201782e+00
 * time: 0.0009999275207519531
     1     1.473289e+03     1.560535e+01
 * time: 0.0019998550415039062
     2     1.442079e+03     4.681610e+01
 * time: 0.003000020980834961
     3     1.348446e+03     1.404482e+02
 * time: 0.004999876022338867
     4     1.067550e+03     2.962982e+02
 * time: 0.006000041961669922
     5     4.749537e+02     3.450557e+01
 * time: 0.007999897003173828
     6     4.059426e+02     3.511205e+01
 * time: 0.009999990463256836
     7     3.357185e+02     6.064606e-01
 * time: 0.013000011444091797
     8     3.345056e+02     8.778015e+00
 * time: 0.018999814987182617
     9     3.169496e+02     8.171539e+00
 * time: 0.020999908447265625
    10     3.006065e+02     1.588013e+00
 * time: 0.021999835968017578
    11     3.006065e+02     5.784607e-02
 * time: 0.023999929428100586
    12     3.004908e+02     1.371613e-01
 * time: 0.026000022888183594
    13     3.002165e+02     7.929993e-02
 * time: 0.029999971389770508
    14     3.000579e+02     2.355957e-02
 * time: 0.03399991989135742
    15     3.000107e+02     2.166748e-03
 * time: 0.03600001335144043
    16     3.000064e+02     2.151543e-03
 * time: 0.03699994087219238
    17     3.000021e+02     5.342755e-04
 * time: 0.03900003433227539
    18     3.000010e+02     4.882812e-04
 * time: 0.039999961853027344
    19     3.000001e+02     9.155273e-05
 * time: 0.04199981689453125
    20     3.000001e+02     3.051758e-05
 * time: 0.046000003814697266
    21     3.000000e+02     0.000000e+00
 * time: 0.04999995231628418
ApplicationDrivenLearning.Solution(300.0f0, Real[20.0f0])
```

### Pros
- Is the most general option as it can be used with basically any plan, assess and forecast model.
- Does not require the gradient of the objective function.

### Cons
- Relies on a lot of iterations to accurately optimize the model parameters.
- Depends on the initial guess that can be given by the user or generated randomly by the Optim.jl package.

## Gradient mode

By computing the gradient of the assessed cost with respect to the forecast values, this mode propagates the gradient to the predictive model parameters, guiding the parameter update process. Since it uses the model structure end-to-end to guide training, it typically requires fewer iterations to achieve good results. However, it may suffer from known issues of gradient-based optimization methods, such as sensitivity to learning rate selection.

The gradient algorithm runs for a specified number of iterations, saving the best available
parameter values until the end or convergence.

### Arguments

- `rule`: The optimizer for the gradient algorithm. Has to be an instance of optimization
rules from Flux.jl.
- `epochs`: The number of iterations to run the gradient algorithm.
- `batch_size`: The batch size to use for the gradient algorithm. If `-1`, the entire dataset is used.
- `verbose`: Whether to print the progress of the gradient algorithm.
- `compute_cost_every`: Allows for cost computation of every sample to be run only
after a specified number of epochs. This enables faster iterations with the drawback of
possibly missing sets of parameters with low associated cost.
- `time_limit`: The time limit for the gradient algorithm in seconds.
- `g_tol`: Convergence condition on the infinite norm of the gradient vector. Below, we illustrate the use of NelderMeadMode to optimize the predictive model used in the ongoing example.

### Example

```julia
julia> opt = ApplicationDrivenLearning.Options(
    ApplicationDrivenLearning.GradientMode,
    rule = Flux.Descent(0.01),
);
julia> sol = ApplicationDrivenLearning.train!(model, X, Y, opt)
Epoch 1 | Time = 0.0s | Cost = 1411.12
Epoch 2 | Time = 0.0s | Cost = 1330.12
Epoch 3 | Time = 0.1s | Cost = 1249.12
Epoch 4 | Time = 0.1s | Cost = 1168.12
Epoch 5 | Time = 0.1s | Cost = 1087.12
Epoch 6 | Time = 0.1s | Cost = 1006.12
Epoch 7 | Time = 0.2s | Cost = 925.12
Epoch 8 | Time = 0.2s | Cost = 844.12
Epoch 9 | Time = 0.2s | Cost = 763.12
Epoch 10 | Time = 0.2s | Cost = 682.12
Epoch 11 | Time = 0.2s | Cost = 601.12
Epoch 12 | Time = 0.3s | Cost = 573.37
Epoch 13 | Time = 0.3s | Cost = 564.37
Epoch 14 | Time = 0.3s | Cost = 555.37
Epoch 15 | Time = 0.3s | Cost = 546.37
Epoch 16 | Time = 0.4s | Cost = 537.37
Epoch 17 | Time = 0.4s | Cost = 528.37
Epoch 18 | Time = 0.4s | Cost = 519.37
Epoch 19 | Time = 0.4s | Cost = 510.37
Epoch 20 | Time = 0.5s | Cost = 501.37
Epoch 21 | Time = 0.5s | Cost = 492.37
Epoch 22 | Time = 0.5s | Cost = 483.37
Epoch 23 | Time = 0.5s | Cost = 474.37
Epoch 24 | Time = 0.5s | Cost = 465.37
Epoch 25 | Time = 0.6s | Cost = 456.37
Epoch 26 | Time = 0.6s | Cost = 447.37
Epoch 27 | Time = 0.6s | Cost = 438.37
Epoch 28 | Time = 0.7s | Cost = 429.37
Epoch 29 | Time = 0.7s | Cost = 420.37
Epoch 30 | Time = 0.7s | Cost = 411.37
Epoch 31 | Time = 0.7s | Cost = 402.37
Epoch 32 | Time = 0.7s | Cost = 393.37
Epoch 33 | Time = 0.8s | Cost = 384.37
Epoch 34 | Time = 0.8s | Cost = 375.37
Epoch 35 | Time = 0.8s | Cost = 366.37
Epoch 36 | Time = 0.8s | Cost = 357.37
Epoch 37 | Time = 0.9s | Cost = 348.37
Epoch 38 | Time = 0.9s | Cost = 339.37
Epoch 39 | Time = 0.9s | Cost = 330.37
Epoch 40 | Time = 0.9s | Cost = 321.37
Epoch 41 | Time = 1.0s | Cost = 312.38
Epoch 42 | Time = 1.0s | Cost = 303.38
Epoch 43 | Time = 1.0s | Cost = 305.62
Epoch 44 | Time = 1.0s | Cost = 303.38
Epoch 45 | Time = 1.1s | Cost = 305.62
Epoch 46 | Time = 1.1s | Cost = 303.38
Epoch 47 | Time = 1.1s | Cost = 305.62
Epoch 48 | Time = 1.1s | Cost = 303.38
Epoch 49 | Time = 1.2s | Cost = 305.62
Epoch 50 | Time = 1.2s | Cost = 303.38
Epoch 51 | Time = 1.2s | Cost = 305.62
Epoch 52 | Time = 1.2s | Cost = 303.38
Epoch 53 | Time = 1.3s | Cost = 305.62
Epoch 54 | Time = 1.3s | Cost = 303.38
Epoch 55 | Time = 1.3s | Cost = 305.62
Epoch 56 | Time = 1.3s | Cost = 303.38
Epoch 57 | Time = 1.4s | Cost = 305.62
Epoch 58 | Time = 1.4s | Cost = 303.38
Epoch 59 | Time = 1.4s | Cost = 305.62
Epoch 60 | Time = 1.4s | Cost = 303.38
Epoch 61 | Time = 1.5s | Cost = 305.62
Epoch 62 | Time = 1.5s | Cost = 303.38
Epoch 63 | Time = 1.5s | Cost = 305.62
Epoch 64 | Time = 1.5s | Cost = 303.38
Epoch 65 | Time = 1.6s | Cost = 305.62
Epoch 66 | Time = 1.6s | Cost = 303.38
Epoch 67 | Time = 1.6s | Cost = 305.62
Epoch 68 | Time = 1.6s | Cost = 303.38
Epoch 69 | Time = 1.7s | Cost = 305.62
Epoch 70 | Time = 1.7s | Cost = 303.38
Epoch 71 | Time = 1.7s | Cost = 305.62
Epoch 72 | Time = 1.7s | Cost = 303.38
Epoch 73 | Time = 1.8s | Cost = 305.62
Epoch 74 | Time = 1.8s | Cost = 303.38
Epoch 75 | Time = 1.8s | Cost = 305.62
Epoch 76 | Time = 1.8s | Cost = 303.38
Epoch 77 | Time = 1.9s | Cost = 305.62
Epoch 78 | Time = 1.9s | Cost = 303.38
Epoch 79 | Time = 1.9s | Cost = 305.62
Epoch 80 | Time = 1.9s | Cost = 303.38
Epoch 81 | Time = 2.0s | Cost = 305.62
Epoch 82 | Time = 2.0s | Cost = 303.38
Epoch 83 | Time = 2.0s | Cost = 305.62
Epoch 84 | Time = 2.0s | Cost = 303.38
Epoch 85 | Time = 2.1s | Cost = 305.62
Epoch 86 | Time = 2.1s | Cost = 303.38
Epoch 87 | Time = 2.1s | Cost = 305.62
Epoch 88 | Time = 2.1s | Cost = 303.38
Epoch 89 | Time = 2.2s | Cost = 305.62
Epoch 90 | Time = 2.2s | Cost = 303.38
Epoch 91 | Time = 2.2s | Cost = 305.62
Epoch 92 | Time = 2.2s | Cost = 303.38
Epoch 93 | Time = 2.3s | Cost = 305.62
Epoch 94 | Time = 2.3s | Cost = 303.38
Epoch 95 | Time = 2.3s | Cost = 305.62
Epoch 96 | Time = 2.3s | Cost = 303.38
Epoch 97 | Time = 2.4s | Cost = 305.62
Epoch 98 | Time = 2.4s | Cost = 303.38
Epoch 99 | Time = 2.4s | Cost = 305.62
Epoch 100 | Time = 2.4s | Cost = 303.38
ApplicationDrivenLearning.Solution(303.3750343322754, Real[19.887499f0])
```

### Pros

- Tends to require fewer iterations to achieve good results.
- Can use different optimizers from Flux.jl and strategies such as stochastic gradient descent with small batches.

### Cons

- Relies on the gradient of the objective function, which can be costly to compute, sensitive to the learning rate and on the initial guess.
- Does not support integer variables in the plan and assess models.