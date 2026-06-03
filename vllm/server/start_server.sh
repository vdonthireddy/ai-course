#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo "Starting vLLM server with google/gemma-4-e2b-it on Metal..."
echo "=========================================================="

# Resolve the project root and look for common virtual environment paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VENV_PATHS=(
    "$SCRIPT_DIR/.venv-vllm/bin/activate"
    "$SCRIPT_DIR/../../.venv-vllm/bin/activate"
    "$HOME/.venv-vllm-metal/bin/activate"
)

VENV_ACTIVATED=false
for path in "${VENV_PATHS[@]}"; do
    if [ -f "$path" ]; then
        echo "Found virtual environment: $path"
        echo "Activating virtual environment..."
        source "$path"
        VENV_ACTIVATED=true
        break
    fi
done

if [ "$VENV_ACTIVATED" = false ]; then
    echo "WARNING: No virtual environment (.venv-vllm or ~/.venv-vllm-metal) found."
    echo "Attempting to run using current system shell environment..."
fi

# Set local model directory path
MODEL_DIR="$SCRIPT_DIR/../models/gemma-4-e2b-it"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Local model directory not found at: $MODEL_DIR"
    echo "To download the model weights to the local directory (minimizing runtime HF dependency), run:"
    echo "  hf download google/gemma-4-e2b-it --local-dir ./vllm/models/gemma-4-e2b-it"
    exit 1
fi

# Set vLLM to run in offline mode
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "Enforcing offline mode (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1)"

# Set vLLM Apple Silicon specific environment variables
# VLLM_METAL_MEMORY_FRACTION controls the fraction of unified memory allocated to the vLLM engine.
export VLLM_METAL_MEMORY_FRACTION=0.95
echo "Setting VLLM_METAL_MEMORY_FRACTION=0.95"

# Serve the model from the local directory
echo "Launching vLLM service from local folder..."
exec vllm serve "$MODEL_DIR" \
    --served-model-name google/gemma-4-e2b-it \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 2048 \
    --enable-prefix-caching \
    --trust-remote-code
