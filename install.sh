#!/bin/bash

# Self-fix permissions if the user forgot to chmod +x
chmod +x "$0" 2>/dev/null || true

# =============================================================================
# LoRA Captioner - One-time Installation Script
# =============================================================================
# This script sets up a Python virtual environment and installs all
# dependencies so the tool is ready to use after cloning the repo.
#
# Usage (recommended):
#   ./install.sh
# =============================================================================

set -e

echo "=========================================="
echo "  LoRA Captioner - Installer"
echo "=========================================="
echo ""

# Detect OS family
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS_ID=$ID
    OS_LIKE=${ID_LIKE:-}
else
    OS_ID=$(uname -s)
    OS_LIKE=""
fi

# Function to install Python if missing
install_python() {
    echo "Python 3 not found. Attempting to install..."

    if command -v apt &> /dev/null; then
        echo "Detected apt-based system (Ubuntu/Debian/WSL)"
        sudo apt update
        sudo apt install -y python3 python3-venv python3-pip

    elif command -v dnf &> /dev/null; then
        echo "Detected dnf-based system (Fedora/RHEL/Rocky)"
        sudo dnf install -y python3 python3-venv python3-pip

    elif command -v yum &> /dev/null; then
        echo "Detected yum-based system (older RHEL/CentOS)"
        sudo yum install -y python3 python3-pip
        # venv is often included or needs python3-devel

    elif command -v pacman &> /dev/null; then
        echo "Detected pacman-based system (Arch/Manjaro)"
        sudo pacman -S --noconfirm python python-pip

    elif command -v zypper &> /dev/null; then
        echo "Detected zypper-based system (openSUSE)"
        sudo zypper install -y python3 python3-venv python3-pip

    else
        echo ""
        echo "ERROR: Could not automatically detect a supported package manager."
        echo ""
        echo "Please install Python 3 + venv support manually, then re-run this script."
        echo ""
        echo "Common commands by distribution:"
        echo "  Ubuntu/Debian/WSL:   sudo apt install python3 python3-venv python3-pip"
        echo "  Fedora/RHEL:         sudo dnf install python3 python3-venv python3-pip"
        echo "  Arch/Manjaro:        sudo pacman -S python python-pip"
        echo "  openSUSE:            sudo zypper install python3 python3-venv python3-pip"
        echo ""
        exit 1
    fi
}

# Check for python3
if ! command -v python3 &> /dev/null; then
    install_python
fi

# Final verification
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is still not available after installation attempt."
    echo "Please install it manually and re-run this script."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"

# Create virtual environment
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists at $VENV_DIR"
else
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR" || {
        echo ""
        echo "Failed to create virtual environment."
        echo "On some systems you may need to install the 'python3-venv' package."
        echo ""
        echo "Try: sudo apt install python3-venv   (or equivalent for your distro)"
        exit 1
    }
    echo "Virtual environment created successfully."
fi

# Activate and install dependencies
echo ""
echo "Activating virtual environment and installing dependencies..."

source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "To use the tool from now on, run:"
echo ""
echo "  cd ~/lora-captioner"
echo "  source .venv/bin/activate"
echo "  python lora_captioner.py --help"
echo ""
echo "Example:"
echo "  python lora_captioner.py --backend ollama --dry-run --limit 5"
echo ""
echo "The virtual environment is now ready."
echo ""
