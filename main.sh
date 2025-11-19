#!/bin/bash

# ============================================================================
# Main script to run all experiments
# ============================================================================
# This script orchestrates the execution of all setup and experiment scripts
# in the correct order.

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ============================================================================
# Setup phase
# ============================================================================
echo "=========================================="
echo "Starting Application-Driven Learning"
echo "Experiments Setup and Execution"
echo "=========================================="
echo ""

# Run Julia setup
echo "[INFO] Running Julia setup..."
bash julia_setup.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] Julia setup failed."
    exit 1
fi

# # Run Python setup
# echo ""
# echo "[INFO] Running Python setup..."
# bash python_setup.sh
# if [ $? -ne 0 ]; then
#     echo "[ERROR] Python setup failed."
#     exit 1
# fi

# ============================================================================
# Experiments phase
# ============================================================================
echo ""
echo "=========================================="
echo "Starting Experiments"
echo "=========================================="

# # Run Newsvendor experiments
# echo ""
# echo "[INFO] Running Newsvendor experiments..."
# bash newsvendor_experiments.sh
# if [ $? -ne 0 ]; then
#     echo "[ERROR] Newsvendor experiments failed."
#     exit 1
# fi

# Run Knapsack experiments
echo ""
echo "[INFO] Running Knapsack experiments..."
bash knapsack_experiment.sh
if [ $? -ne 0 ]; then
    echo "[ERROR] Knapsack experiments failed."
    exit 1
fi

# # Run Shortest Path experiments
# echo ""
# echo "[INFO] Running Shortest Path experiments..."
# bash shortest_path_experiment.sh
# if [ $? -ne 0 ]; then
#     echo "[ERROR] Shortest Path experiments failed."
#     exit 1
# fi

# # Run Matpower experiments
# echo ""
# echo "[INFO] Running Matpower experiments..."
# bash matpower_experiment.sh
# if [ $? -ne 0 ]; then
#     echo "[ERROR] Matpower experiments failed."
#     exit 1
# fi

# # Deactivate virtual environment if it was activated
# if [ -n "$VIRTUAL_ENV" ]; then
#     echo ""
#     echo "[INFO] Deactivating virtual environment..."
#     deactivate
# fi

# # ============================================================================
# # Success message
# # ============================================================================
# echo ""
# echo "=========================================="
# echo "[SUCCESS] All experiments completed."
# echo "=========================================="
# echo ""
