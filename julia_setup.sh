#!/bin/bash

# ============================================================================
# Check and install Julia if not present
# ============================================================================
echo "=== Checking Julia Installation ==="
if ! command -v julia &> /dev/null; then
    echo "[INFO] Julia not found. Installing Julia..."
    # Install Julia using official method for Amazon Linux
    curl -fsSL https://install.julialang.org | sh
    # Add Julia to PATH for current session
    export PATH="$HOME/.juliaup/bin:$PATH"
    # Verify installation
    if ! command -v julia &> /dev/null; then
        echo "[ERROR] Julia installation failed. Please install manually."
        exit 1
    fi
    echo "[SUCCESS] Julia installed successfully."
else
    echo "[INFO] Julia is already installed: $(julia --version)"
fi

