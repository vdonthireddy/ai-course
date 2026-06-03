# Serving Gemma 4 (2B) on Apple Silicon using vLLM Metal

This guide provides step-by-step instructions for downloading and setting up the **google/gemma-4-e2b-it** model and the **vllm-metal** inference engine to serve OpenAI-compatible endpoints on Apple Silicon (M-series) Macs.

We provide automation scripts in the server and client directories to simplify the process:
*   [start_server.sh](file:///Users/donthireddy/code/ai-course/vllm/server/start_server.sh): Script to configure memory environment variables and launch the vLLM server.
*   [test_client.py](file:///Users/donthireddy/code/ai-course/vllm/client/test_client.py): Python script using the OpenAI SDK to run a test completion with streaming.

---

## Overview

**vLLM Metal** is a community-maintained hardware plugin for vLLM that enables native, high-performance LLM inference on macOS using Apple's **MLX** framework as the compute backend. By leveraging MLX and Apple Silicon’s **Unified Memory Architecture**, `vllm-metal` achieves zero-copy memory transfers and accelerated token generation.

We focus here on **google/gemma-4-e2b-it** (Effective 2B parameters, Instruction-Tuned). It is optimized for mobile, edge, and local developer workflows, boasting reasoning capabilities and multimodality within a lightweight memory footprint.

---

## Prerequisites

Before proceeding, ensure your system meets the following requirements:

1. **Hardware**: Apple Silicon Mac (M1, M2, M3, M4 series).
2. **Unified Memory (RAM)**: 8 GB or more (16 GB recommended to allow headroom for OS operations).
3. **Operating System**: macOS Sonoma (14.0) or later.
4. **Python Environment**: Native **arm64 Python 3.12** installed.
   > [!WARNING]
   > Rosetta or x86_64 translated Python installations are **not supported** and will fail when compiling or loading MLX extensions.
5. **Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```
6. **Hugging Face Account**: Required to accept the Gemma 4 terms of use and obtain an access token.

---

## Step 1: Environment Setup

Using a virtual environment prevents conflicts between vLLM's dependencies and other Python packages. You can create one using the standard Python `venv` library or the fast package manager `uv`.

### Option A: Using `uv` (Recommended)
`uv` is extremely fast and resolves vLLM's dependencies cleanly.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate a Python 3.12 virtual environment in the project root
uv venv .venv-vllm --python 3.12
source .venv-vllm/bin/activate
```

### Option B: Using Python `venv`
```bash
python3.12 -m venv .venv-vllm
source .venv-vllm/bin/activate
```

---

## Step 2: Install `vllm-metal`

There are two primary ways to install the inference engine:

### Option A: Using the Official Installer Script (Easiest)
The vLLM project provides a single command script to set up `vllm-metal` and its core dependencies:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
```
*By default, this script creates and installs vLLM into `~/.venv-vllm-metal`. To activate it, run:*
```bash
source ~/.venv-vllm-metal/bin/activate
```

### Option B: Manual Installation via pip
If you prefer to install directly into your active virtual environment:

```bash
pip install --upgrade pip
pip install vllm-metal
```

---

## Step 3: One-Time Model Download (Minimizing Runtime HF Dependency)

To run vLLM completely offline and avoid checking or connecting to Hugging Face during server execution, perform a one-time download of the model weights to a local directory. Run these commands from the project root (`/Users/donthireddy/code/ai-course`):

1. Visit [Hugging Face](https://huggingface.co/google/gemma-4-e2b-it) and accept the Gemma 4 license terms.
2. Generate a **User Access Token** (Read permission) under **Settings -> Access Tokens** on Hugging Face.
3. Authenticate your terminal using `hf` (since `huggingface-cli` is deprecated on your system):
   ```bash
   hf auth login
   ```
4. Download the model weights directly to the local directory in the repository:
   ```bash
   hf download google/gemma-4-e2b-it --local-dir ./vllm/models/gemma-4-e2b-it
   ```

---

## Step 4: Run the API Server

We provide a launcher script that handles system path activation, sets Apple-Silicon-specific memory parameters, and executes the server.

### Memory Allocation on Apple Silicon
> [!IMPORTANT]
> The standard vLLM flag `--gpu-memory-utilization` is **ignored** by the Metal plugin. To control memory allocation, you must set the environment variable `VLLM_METAL_MEMORY_FRACTION`.
> Our launcher script sets `VLLM_METAL_MEMORY_FRACTION=0.95` (allocating 95% of your unified memory limit to the engine) and adds `--max-model-len 2048`. These settings are required to fit the model's footprint and leaves sufficient budget for the Key-Value (KV) cache on 16 GB systems.

### Starting the Server in Offline Mode
Ensure you have downloaded the weights to `../models/gemma-4-e2b-it` as shown in Step 3.

Run the startup script in your terminal:
```bash
./start_server.sh
```

> [!NOTE]
> The `start_server.sh` script automatically sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. This forces vLLM to run in **offline mode**, preventing any checks or queries to Hugging Face at runtime, and loads weights directly from your local `./vllm/models/gemma-4-e2b-it` directory.

Once you see the output `Uvicorn running on http://0.0.0.0:8000`, the server is ready.

---

## Step 5: Test and Interact

While the server is running in one terminal tab, open another tab and run our Python test script.

### Using the Python Test Client
Make sure the `openai` SDK is installed, then run the script:

```bash
# Install the client library if needed
pip install openai

# Navigate to the client directory and run the test client
cd ../client
python3 test_client.py
```
This script queries your local server and streams the response for the prompt *"Explain the difference between prefill and decode stages in LLM inference"* back to your terminal.

### Using cURL
Alternatively, you can test the OpenAI-compatible endpoint directly via `curl`:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/gemma-4-e2b-it",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain what PagedAttention is in two sentences."}
    ],
    "temperature": 0.7
  }'
```

---

## Advanced Configurations

*   **`--kv-cache-dtype`**: To further optimize memory consumption, you can specify FP8 format for the key-value cache (e.g. `vllm serve google/gemma-4-e2b-it --kv-cache-dtype fp8`).
*   **Swapping and Preemption**: If the system's available memory becomes constrained during massive concurrent requests, vLLM's Block Manager will automatically preempt sequences and swap block cache pages to CPU/system swap space to maintain system stability.
*   **LAN Exposure**: The server binds to `--host 0.0.0.0` by default, meaning any device on your local network can access the model endpoint at `http://<your-macbook-ip>:8000/v1`.
