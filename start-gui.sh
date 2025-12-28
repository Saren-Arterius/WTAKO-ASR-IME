#!/bin/bash

# GLM-ASR-STT Launch Script

# Ensure we are in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        echo "Virtual environment not found. Please run ./setup.sh first."
        exit 1
    fi
fi

# Function to print usage
print_usage() {
    echo "Usage: $0 [option]"
    echo "Options:"
    echo "  --gfx VER   Set HSA_OVERRIDE_GFX_VERSION=VER and launch (e.g., --gfx 10.3.5)"
    echo "  --help      Show this help message"
}

# Check for GFX version override
if [ "$1" == "--gfx" ]; then
    if [ -z "$2" ]; then
        echo "Error: --gfx requires a version number."
        exit 1
    fi
    export HSA_OVERRIDE_GFX_VERSION=$2
    echo "HSA_OVERRIDE_GFX_VERSION set to $HSA_OVERRIDE_GFX_VERSION"
fi

echo "Launching GUI..."
uv run client/gui.py
