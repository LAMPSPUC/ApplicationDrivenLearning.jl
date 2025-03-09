"""
    Solution

A struct to store the result of the optimisation process with final cost and solution.
"""
struct Solution
    cost::Real
    params::Vector{Real}
end
