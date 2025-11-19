#!/bin/bash

# ============================================================================
# Check and install Python3 if not present
# ============================================================================
echo "=== Checking Python Installation ==="
if ! command -v python3 &> /dev/null; then
    echo "[INFO] Python3 not found. Installing Python3..."
    # Install Python3 (Amazon Linux 2023 uses dnf)
    if command -v dnf &> /dev/null; then
        sudo dnf install -y python3 python3-pip
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3 python3-pip
    else
        echo "[ERROR] Cannot find package manager (dnf/yum). Please install Python3 manually."
        exit 1
    fi
    # Verify installation
    if ! command -v python3 &> /dev/null; then
        echo "[ERROR] Python3 installation failed. Please install manually."
        exit 1
    fi
    echo "[SUCCESS] Python3 installed successfully."
else
    echo "[INFO] Python3 is already installed: $(python3 --version)"
fi

# ============================================================================
# Python virtual environment setup
# ============================================================================
echo ""
echo "=== Python Environment Setup ==="
echo "[INFO] Creating Python virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "[INFO] Virtual environment created."
else
    echo "[INFO] Virtual environment already exists."
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "[INFO] Installing Python dependencies..."
pip install --upgrade pip > /dev/null 2>&1
pip install -r experiments/knapsack/python/requirements.txt
echo "[SUCCESS] Python dependencies installed."

