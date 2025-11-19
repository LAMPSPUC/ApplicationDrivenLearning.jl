#!/bin/bash

# ============================================================================
# Run Julia Newsvendor experiments
# ============================================================================
echo ""
echo "=== Starting Julia Newsvendor Experiments ==="
echo ""

# Run newsvendor_2.jl script
echo "[INFO] Running script: experiments/newsvendor_2/newsvendor_2.jl"
julia --project=. experiments/newsvendor_2/newsvendor_2.jl

# Run newsvendor_3.jl script
echo "[INFO] Running script: experiments/newsvendor_3/newsvendor_3.jl"
julia --project=. experiments/newsvendor_3/newsvendor_3.jl

# Run post-analysis for newsvendor_3
echo ""
echo "=== Running Newsvendor 3 Post-Analysis ==="
echo "[INFO] Running script: experiments/newsvendor_3/post_analysis.jl"
julia --project=. experiments/newsvendor_3/post_analysis.jl

echo "[SUCCESS] Newsvendor experiments completed."

