# CLAUDE.md

## Project Overview

This is a deployment project for SGLang (https://github.com/sgl-project/sglang), a high-performance serving framework for large language models. The primary goal is to deploy and run large LLMs like Llama 3 70B that require distributed inference across multiple H100 GPUs.

## Context

- **Development Environment**: macOS (MacBook Pro)
- **Deployment Target**: Slurm cluster with H100 GPUs
- **Primary Use Case**: Running Llama 3 70B with tensor parallelism across multiple GPUs

## Architecture

### SGLang Overview
SGLang is a fast serving framework that provides:
- Low-latency, high-throughput inference
- RadixAttention prefix caching
- Support for tensor/pipeline/expert/data parallelism
- Extensive model support (Llama, Qwen, DeepSeek, etc.)
- Hardware compatibility across NVIDIA, AMD, Intel, Google TPUs

### Parallelism Strategies

For large models like Llama 3 70B that don't fit on a single GPU:

1. **Tensor Parallelism (TP)**: Splits model layers across multiple GPUs
   - Best for models that don't fit in single GPU memory
   - Higher communication overhead (requires fast interconnect)
   - Use `--tp N` flag

2. **Data Parallelism (DP)**: Replicates model across GPU groups
   - Better for throughput when memory allows
   - Can be combined with TP: `--tp 2 --dp 2` (uses 4 GPUs total)

3. **Pipeline Parallelism (PP)**: Distributes model layers for pipeline execution
   - Use `--pp N` for pipeline stages

## Key SGLang Parameters

### Essential Flags
- `--model-path`: Model location (HuggingFace or local)
- `--tp N`: Tensor parallelism across N GPUs
- `--host`: Server host (0.0.0.0 for external access)
- `--port`: Server port (default 30000)
- `--mem-fraction-static`: KV cache memory fraction (default 0.9)
- `--trust-remote-code`: Allow custom HuggingFace models

### Multi-Node Setup
- `--nnodes N`: Number of nodes
- `--node-rank R`: Current node rank (0 for master)
- `--dist-init-addr`: Distributed initialization address

### Performance Tuning
- `--dtype`: Model precision (auto, half, bfloat16, float32)
- `--quantization`: Quantization method (awq, fp8, gptq, etc.)
- `--attention-backend`: Attention kernel (flashinfer, triton, etc.)
- `--enable-torch-compile`: PyTorch compilation for speedup

## Deployment Notes

### Llama 3 70B Requirements
- Estimated memory: ~140GB in FP16, ~70GB in FP8
- Recommended: 4-8 H100 GPUs (80GB each) with tensor parallelism
- Single node with 8x H100 (80GB) is sufficient for FP16
- Can use 4x H100 with FP8 quantization

### Slurm Considerations
- Use `--gres=gpu:h100:N` to request H100 GPUs
- Set appropriate time limits for long-running inference servers
- Configure `CUDA_VISIBLE_DEVICES` for multi-GPU setups
- Ensure high-speed interconnect (NVLink, InfiniBand) for TP

## Development Workflow

1. Development/testing on MacBook (no GPU, using CPU mode or small models)
2. Deploy to Slurm cluster for production inference
3. Use Makefile targets for common operations
4. Monitor with SGLang metrics and health endpoints

## Useful Resources

- SGLang Docs: https://docs.sglang.ai/
- GitHub: https://github.com/sgl-project/sglang
- Server Arguments: https://docs.sglang.ai/advanced_features/server_arguments.html
- Multi-Node Deployment: Check SGLang docs for latest guidance
