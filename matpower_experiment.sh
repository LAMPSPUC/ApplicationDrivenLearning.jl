#!/bin/bash

# ============================================================================
# Run Matpower experiments
# ============================================================================
echo ""
echo "=== Running Matpower Experiments ==="
echo "[INFO] Running script: experiments/matpower/auto_run.jl"
julia --project=. experiments/matpower/auto_run.jl

echo "[SUCCESS] Matpower experiments completed."

