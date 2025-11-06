#!/bin/bash
#
# CoreWeave B200 Cluster - SGLang Llama 3 70B Deployment
#
# COMPUTING ENVIRONMENT:
# - Cluster: CoreWeave sunk.poc035-usw01a.coreweave.app
# - Hardware: 8x NVIDIA B200 GPUs per node (183GB memory each)
# - CUDA: 12.9
# - Purpose: Test if NCCL collective sync errors from H200 cluster are:
#   1. Cluster-specific (H200 infrastructure issue) OR
#   2. SGLang bug (happens on both clusters)
#
# RATIONALE FOR SEPARATE FILE:
# - Different cluster configuration (B200 vs H200)
# - Different partition names (b200-low vs h200-reserved)
# - No node exclusions needed (clean cluster, no known NCCL issues)
# - May need different NCCL environment variables
# - Keeps H200 and CoreWeave configurations separate for comparison
#
# SETUP APPROACH:
# - Dependencies installed on compute node (not login node)
# - Home directory is shared across all nodes
# - First run will install uv + packages, subsequent runs use cached .venv
#

#SBATCH --job-name=sglang-llama3-70b-coreweave
#SBATCH --output=logs/sglang-coreweave-%j.out
#SBATCH --error=logs/sglang-coreweave-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=8
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --partition=b200-low

# Exit on error
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

echo "=========================================="
echo "CoreWeave B200 Cluster - SGLang Test"
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "=========================================="

# Set up environment paths
PROJECT_DIR=${PROJECT_DIR:-$HOME/sglang-deployment-poc}
cd $PROJECT_DIR

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Ensure uv is in PATH (both possible install locations)
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Install/sync dependencies if .venv doesn't exist or is incomplete
if [ ! -d ".venv" ] || [ ! -f ".venv/pyvenv.cfg" ]; then
    echo "Installing dependencies with uv (first run)..."
    uv sync
else
    echo "Using existing virtual environment..."
fi

# Activate the virtual environment
source .venv/bin/activate

# Set environment variables for all 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL configuration - start with minimal config
# Will add CoreWeave-specific settings if needed based on test results
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# HuggingFace configuration
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_TOKEN=${HF_TOKEN:-""}  # Set in .env file

# Load HF_TOKEN from .env if it exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep HF_TOKEN | xargs)
fi

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
echo "  Python: $(which python)"
echo "  CUDA Devices: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# Launch SGLang server with tensor parallelism
# CUDA graphs disabled - same as final H200 configuration for direct comparison
echo "Launching SGLang server..."
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
