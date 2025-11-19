#!/bin/bash

# ============================================================================
# Run Shortest Path experiments (Python and Julia)
# ============================================================================
echo ""
echo "=== Running Shortest Path Experiments ==="

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

# Run Python shortest_path experiments
cd experiments/shortest_path/python
echo "[INFO] Generating data for shortest_path..."
python generate_data.py
echo "[INFO] Running shortest_path pyepo script..."
python run_pyepo.py
cd ../../..

# Run Julia shortest_path experiments
echo ""
echo "=== Running Julia Shortest Path Experiments ==="
echo "[INFO] Running script: experiments/shortest_path/julia/shortest_path.jl"
julia --project=. experiments/shortest_path/julia/shortest_path.jl

# Run post-analysis
echo ""
echo "=== Running Shortest Path Post-Analysis ==="
echo "[INFO] Running script: experiments/shortest_path/julia/post_analysis.jl"
julia --project=. experiments/shortest_path/julia/post_analysis.jl

echo "[SUCCESS] Shortest Path experiments completed."

