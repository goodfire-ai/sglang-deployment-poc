# SGLang Deployment

High-performance LLM inference deployment using [SGLang](https://github.com/sgl-project/sglang) on Slurm clusters with H100 GPUs.

## Overview

This project provides infrastructure for deploying large language models (especially Llama 3 70B) using SGLang's distributed inference capabilities. It includes:

- Python project configuration with uv and lockfile
- Makefile for common operations
- Slurm batch scripts for cluster deployment
- Support for tensor parallelism across multiple GPUs
- Interactive chat client for testing

## Prerequisites

- **Development**: Python 3.10+, uv package manager
- **Deployment**: Slurm cluster with H100 GPUs, CUDA 12.1+

## Quick Start

### 1. Environment Setup

First, configure your environment variables:

```bash
# Create .env from template
make setup

# Edit .env and add your HuggingFace token
# Get token from: https://huggingface.co/settings/tokens
vi .env  # or use your preferred editor

# Validate configuration
make validate
```

### 2. Installation

```bash
# Install dependencies using uv
make install

# Or manually
uv sync
```

### 3. Deploy & Test

```bash
# Submit Slurm job for Llama 3 70B on 4-8 H100 GPUs
sbatch slurm/llama3-70b-single-node.sbatch

# Once server is running, test with interactive chat
make chat HOST=your-cluster-node PORT=30000
```

## Makefile Targets

- `make setup` - Create .env from .env.example (first time setup)
- `make validate` - Validate environment configuration
- `make install` - Install dependencies from lockfile
- `make lock` - Update uv.lock lockfile
- `make format` - Format code with ruff
- `make lint` - Lint code with ruff
- `make start-local` - Start SGLang server locally
- `make start-server` - Start server with custom parameters
- `make health-check` - Check server health
- `make chat` - Interactive chat with server (test inference speed)
- `make stop-server` - Stop running server
- `make clean` - Clean cache and temporary files

## Configuration

### Environment Variables

The project uses a `.env` file for configuration (not tracked in git). Use `make setup` to create it from the template:

```bash
# Create .env from .env.example
make setup

# Validate your configuration
make validate
```

Key environment variables in `.env`:

```bash
# HuggingFace token (REQUIRED)
# Get from: https://huggingface.co/settings/tokens
HF_TOKEN=your_token_here

# Model path (HuggingFace or local)
MODEL_PATH=meta-llama/Meta-Llama-3-70B-Instruct

# Server configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=30000

# GPU configuration
TENSOR_PARALLEL_SIZE=4
MEM_FRACTION=0.85
```

See `.env.example` for all available configuration options.

### Slurm Configuration

The provided Slurm scripts assume:
- Partition name: `gpu` (adjust in sbatch scripts)
- GPU type: `h100` (adjust if using A100 or others)
- CUDA module: `cuda/12.1` (adjust to your cluster)

## Model Requirements

### Llama 3 70B
- **Memory**: ~140GB (FP16), ~70GB (FP8)
- **Recommended**: 4-8x H100 (80GB) GPUs
- **Tensor Parallelism**: `--tp 4` or `--tp 8`

## Usage Examples

### Start Server Manually

```bash
# Single GPU
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --host 0.0.0.0 \
    --port 30000

# Multi-GPU with tensor parallelism (4 GPUs)
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-70B-Instruct \
    --tp 4 \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static 0.85
```

### Interactive Chat (Hello World Test)

The easiest way to test your server and see inference speed:

```bash
# Start interactive chat (assumes server is running)
make chat

# Or connect to a remote server
make chat HOST=your-server-hostname PORT=30000
```

This opens an interactive chat session where you can:
- Test the server is working correctly
- Get real-time inference speed metrics (tokens/sec)
- Have multi-turn conversations
- Use commands: `/reset` (clear history), `/quit` (exit), `/help` (show help)

Example session:
```
======================================================================
SGLang Interactive Chat
======================================================================

Server: localhost:30000
Model:  meta-llama/Meta-Llama-3-70B-Instruct

Commands:
  /reset  - Clear conversation history
  /quit   - Exit chat
  /help   - Show this help message

======================================================================

You: Hello! Can you explain what tensor parallelism is?