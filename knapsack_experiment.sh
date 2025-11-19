#!/bin/bash

# ============================================================================
# Run Knapsack experiments (Python and Julia)
# ============================================================================
echo ""
echo "=== Running Knapsack Experiments ==="

# Activate virtual environment if not already activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        echo "[INFO] Activating virtual environment..."
        source venv/bin/activate
    else
        echo "[ERROR] Virtual environment not found. Please run python_setup.sh first."
        exit 1
    fi
fi

# Run Python knapsack experiments
cd experiments/knapsack/python
echo "[INFO] Generating data for knapsack..."
python generate_data.py
echo "[INFO] Running knapsack pyepo script..."
python run_pyepo.py
cd ../../..

# Run Julia knapsack experiments
echo ""
echo "=== Running Julia Knapsack Experiments ==="
echo "[INFO] Running script: experiments/knapsack/julia/knapsack.jl"
julia --project=. experiments/knapsack/julia/knapsack.jl

# Run post-analysis
echo ""
echo "=== Running Knapsack Post-Analysis ==="
echo "[INFO] Running script: experiments/knapsack/julia/post_analysis.jl"
julia --project=. experiments/knapsack/julia/post_analysis.jl

echo "[SUCCESS] Knapsack experiments completed."

