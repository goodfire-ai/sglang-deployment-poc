# SGLang Deployment

High-performance LLM inference deployment using [SGLang](https://github.com/sgl-project/sglang) on Slurm clusters with H100 GPUs.

## Overview

This project provides infrastructure for deploying large language models (especially Llama 3 70B) using SGLang's distributed inference capabilities. It includes:

- Python project configuration with uv and lockfile
- Makefile for common operations
- Slurm batch scripts for cluster deployment
- Support for tensor parallelism across multiple GPUs

## Prerequisites

- **Development**: Python 3.10+, uv package manager
- **Deployment**: Slurm cluster with H100 GPUs, CUDA 12.1+

## Quick Start

### Installation

```bash
# Install dependencies using uv
make install

# Or manually
uv sync
```

### Local Development

```bash
# Format code
make format

# Lint code
make lint

# Start server locally (for testing, CPU mode)
make start-local
```

### Cluster Deployment

#### Single-Node Multi-GPU (Llama 3 70B)

```bash
# Submit Slurm job for Llama 3 70B on 4-8 H100 GPUs
sbatch slurm/llama3-70b-single-node.sbatch
```

Edit `slurm/llama3-70b-single-node.sbatch` to customize:
- Number of GPUs (`--gres=gpu:h100:N`)
- Tensor parallelism size (`--tp`)
- Model path
- Memory settings

## Makefile Targets

- `make install` - Install dependencies from lockfile
- `make lock` - Update uv.lock lockfile
- `make format` - Format code with ruff
- `make lint` - Lint code with ruff
- `make start-local` - Start SGLang server locally
- `make start-server` - Start server with custom parameters
- `make health-check` - Check server health
- `make stop-server` - Stop running server
- `make clean` - Clean cache and temporary files

## Configuration

### Environment Variables

Create a `.env` file (not tracked in git):

```bash
# HuggingFace token for model access
HF_TOKEN=your_token_here

# Model path (HuggingFace or local)
MODEL_PATH=meta-llama/Meta-Llama-3-70B-Instruct

# Server configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=30000

# GPU configuration
TENSOR_PARALLEL_SIZE=4
```

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

### Query Server

```bash
# Health check
curl http://localhost:30000/health

# Generate text (OpenAI-compatible API)
curl http://localhost:30000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "prompt": "Once upon a time",
        "max_tokens": 100
    }'
```

## Troubleshooting

### Out of Memory
- Reduce `--mem-fraction-static` (try 0.8 or 0.85)
- Increase tensor parallelism size (`--tp`)
- Use quantization (`--quantization fp8`)

### Slow Performance
- Ensure high-speed GPU interconnect (NVLink)
- Check tensor parallelism isn't too aggressive
- Consider using data parallelism for throughput

### CUDA Errors
- Set `CUDA_HOME` environment variable
- Check CUDA compatibility (requires CUDA 12.1+)
- Try alternative attention backend: `--attention-backend triton`

## Resources

- [SGLang Documentation](https://docs.sglang.ai/)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [Server Arguments Reference](https://docs.sglang.ai/advanced_features/server_arguments.html)

## License

This deployment configuration is provided as-is. See SGLang's license for framework terms.
