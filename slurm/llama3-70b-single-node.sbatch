#!/bin/bash
#SBATCH --job-name=sglang-llama3-70b
#SBATCH --output=logs/sglang-%j.out
#SBATCH --error=logs/sglang-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

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

# Load required modules (adjust to your cluster)
module purge
module load cuda/12.1
module load python/3.10

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
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# HuggingFace configuration
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_TOKEN=${HF_TOKEN:-""}  # Set in your environment or .env file

# Model configuration
MODEL_PATH=${MODEL_PATH:-"meta-llama/Meta-Llama-3-70B-Instruct"}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-4}
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
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --tp $TENSOR_PARALLEL \
    --host 0.0.0.0 \
    --port $SERVER_PORT \
    --mem-fraction-static $MEM_FRACTION \
    --trust-remote-code \
    --log-level info \
    --dtype auto

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
