#!/bin/bash

# =============================================================================
# LoRA Captioner - Uninstaller
# =============================================================================
# This script removes the virtual environment created by install.sh.
#
# Usage:
#   ./uninstall.sh
# =============================================================================

set -e

echo "=========================================="
echo "  LoRA Captioner - Uninstaller"
echo "=========================================="
echo ""

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "No virtual environment found at $VENV_DIR"
    echo "Nothing to uninstall."
    exit 0
fi

echo "This will remove the virtual environment at: $VENV_DIR"
echo ""

read -p "Are you sure you want to continue? [y/N]: " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Uninstall cancelled."
    exit 0
fi

echo "Removing virtual environment..."
rm -rf "$VENV_DIR"

echo ""
echo "Virtual environment removed successfully."
echo ""
echo "You can re-run ./install.sh anytime to set it up again."
echo ""
