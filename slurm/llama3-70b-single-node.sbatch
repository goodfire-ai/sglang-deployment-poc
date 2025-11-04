#!/bin/bash
#SBATCH --job-name=sglang-llama3-70b
#SBATCH --output=logs/sglang-%j.out
#SBATCH --error=logs/sglang-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:8
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --partition=h200-reserved

# Exit on error
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "=========================================="

# Set up CUDA environment (no module system on this cluster)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# Activate virtual environment
# Adjust path to your project directory
PROJECT_DIR=${PROJECT_DIR:-$HOME/sglang-deployment-poc}
cd $PROJECT_DIR

# Install/sync dependencies if not already done
if [ ! -d ".venv" ]; then
    echo "Installing dependencies with uv..."
    uv sync
fi

# Activate the virtual environment
source .venv/bin/activate

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL configuration for cluster InfiniBand (from Andromeda/cluster ops)
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1

export TORCH_DISTRIBUTED_DEBUG=DETAIL

# HuggingFace configuration
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_TOKEN=${HF_TOKEN:-""}  # Set in your environment or .env file

# Model configuration
MODEL_PATH=${MODEL_PATH:-"meta-llama/Meta-Llama-3-70B-Instruct"}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-8}
SERVER_PORT=${SERVER_PORT:-30000}
MEM_FRACTION=${MEM_FRACTION:-0.85}

echo "=========================================="
echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Tensor Parallelism: $TENSOR_PARALLEL"
echo "  Port: $SERVER_PORT"
echo "  Memory Fraction: $MEM_FRACTION"
echo "=========================================="

# Launch SGLang server with tensor parallelism
# CUDA graphs disabled - collective sync issues during graph capture on this cluster
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --tp $TENSOR_PARALLEL \
    --host 0.0.0.0 \
    --port $SERVER_PORT \
    --mem-fraction-static $MEM_FRACTION \
    --trust-remote-code \
    --log-level info \
    --dtype auto \
    --disable-cuda-graph

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
