#!/bin/bash

# GLM-ASR-STT Setup and Launch Script

# Ensure we are in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        echo "Activating existing virtual environment..."
        source .venv/bin/activate
    else
        echo "Creating new virtual environment with uv..."
        uv venv
        source .venv/bin/activate
    fi
fi

# Function to print usage
print_usage() {
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  --nvidia    Install/Sync for NVIDIA (CUDA) and launch"
    echo "  --rocm      Install/Upgrade for AMD (ROCm 6.4) and launch"
    echo "  --gfx VER   Set HSA_OVERRIDE_GFX_VERSION=VER and launch (e.g., --gfx 10.3.5)"
    echo "  --help      Show this help message"
}

# Default behavior if no arguments
if [ $# -eq 0 ]; then
    echo "No option provided. Defaulting to standard 'uv sync'."
    uv sync
    exit 0
fi

case "$1" in
    --nvidia)
        echo "Setting up for NVIDIA (CUDA)..."
        cp pyproject.cuda.toml pyproject.toml
        if [ -f "uv.cuda.lock" ]; then
            cp uv.cuda.lock uv.lock
        fi
        uv sync --extra cuda
        cp uv.lock uv.cuda.lock
        ;;
    --rocm)
        echo "Setting up for AMD (ROCm 6.4)..."
        cp pyproject.rocm.toml pyproject.toml
        if [ -f "uv.rocm.lock" ]; then
            cp uv.rocm.lock uv.lock
        fi
        uv sync --extra rocm
        cp uv.lock uv.rocm.lock
        ;;
    --help)
        print_usage
        ;;
    *)
        echo "Unknown option: $1"
        print_usage
        exit 1
        ;;
esac
