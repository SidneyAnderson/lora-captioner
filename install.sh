#!/bin/bash

# =============================================================================
# LoRA Captioner - One-time Installation Script
# =============================================================================
# This script sets up a Python virtual environment and installs all
# dependencies so the tool is ready to use after cloning the repo.
#
# Usage:
#   chmod +x install.sh
#   ./install.sh
# =============================================================================

set -e

echo "=========================================="
echo "  LoRA Captioner - Installer"
echo "=========================================="
echo ""

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS=$(uname -s)
fi

# Function to install Python if missing
install_python() {
    echo "Python 3 not found. Attempting to install..."
    
    if command -v apt &> /dev/null; then
        echo "Detected apt-based system (Ubuntu/Debian/WSL)"
        sudo apt update
        sudo apt install -y python3 python3-venv python3-pip
    elif command -v dnf &> /dev/null; then
        echo "Detected dnf-based system (Fedora/RHEL)"
        sudo dnf install -y python3 python3-venv python3-pip
    elif command -v pacman &> /dev/null; then
        echo "Detected pacman-based system (Arch)"
        sudo pacman -S --noconfirm python python-pip
    else
        echo "ERROR: Could not detect package manager."
        echo "Please install Python 3 manually and re-run this script."
        exit 1
    fi
}

# Check for python3
if ! command -v python3 &> /dev/null; then
    install_python
fi

# Verify python3 is now available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 installation failed or is not in PATH."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"

# Create virtual environment
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "Virtual environment created."
fi

# Activate venv and install dependencies
echo "Activating virtual environment and installing dependencies..."

source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install project requirements
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "WARNING: requirements.txt not found!"
fi

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "To use the tool, run the following commands:"
echo ""
echo "  cd ~/lora-captioner"
echo "  source .venv/bin/activate"
echo "  python lora_captioner.py --help"
echo ""
echo "Example usage:"
echo "  python lora_captioner.py --backend ollama --dry-run --limit 3"
echo ""
echo "The virtual environment is now ready. Activate it whenever you want to run the tool."
echo ""
