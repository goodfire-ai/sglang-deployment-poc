# SGLang Deployment Runbook

Quick reference guide for deploying and testing SGLang on Slurm clusters.

## Initial Setup (One-Time)

### 1. Clone and Configure

```bash
# Clone the repository
git clone git@github.com:goodfire-ai/sglang-deployment-poc.git
cd sglang-deployment-poc
```

**Intent**: Get the deployment code on your cluster.
**Tests**: Nothing - just setup.

---

### 2. Environment Configuration

```bash
# Create .env from template
make setup
```

**Intent**: Creates your `.env` configuration file from the template.
**Tests**: Checks if `.env` already exists to avoid overwriting.

```bash
# Edit .env and add your HuggingFace token
vi .env
# Set HF_TOKEN=hf_your_token_here
```

**Intent**: Configure your HuggingFace credentials and deployment settings.
**Tests**: Nothing - manual configuration step.

```bash
# Validate your configuration
make validate
```

**Intent**: Validates all environment variables and checks HuggingFace authentication.
**Tests**:
- `.env` file exists
- Required variables are set (especially `HF_TOKEN`)
- HuggingFace token is valid (authenticates with HF API)
- Model access permissions (if model is gated)

---

### 3. Install Dependencies

```bash
# Install dependencies from lockfile
make install
```

**Intent**: Install Python dependencies (sglang, torch, etc.) using uv.
**Tests**: Verifies all packages install successfully from `uv.lock`.

---

## Deployment to Slurm

### 4. Customize Slurm Script (Optional)

```bash
# View the Slurm script
cat slurm/llama3-70b-single-node.sbatch
```

**Intent**: Review Slurm job configuration before submitting.
**Tests**: Nothing - informational only.

**Common customizations**:
- Number of GPUs: `#SBATCH --gres=gpu:h100:4`
- Tensor parallelism: `TENSOR_PARALLEL=4`
- Model path: `MODEL_PATH=meta-llama/Meta-Llama-3-70B-Instruct`
- Time limit: `#SBATCH --time=24:00:00`
- Partition: `#SBATCH --partition=gpu`

---

### 5. Submit Slurm Job

```bash
# Submit the job
sbatch slurm/llama3-70b-single-node.sbatch
```

**Intent**: Launch SGLang server on Slurm cluster with 4-8 H100 GPUs.
**Tests**: Slurm job acceptance (validates Slurm script syntax and resource availability).

```bash
# Check job status
squeue -u $USER
```

**Intent**: Monitor if your job is queued, running, or completed.
**Tests**: Job scheduling and execution status.

```bash
# View job output (replace JOBID with your job number)
tail -f logs/sglang-JOBID.out
```

**Intent**: Monitor server startup logs in real-time.
**Tests**:
- Model loading (downloads from HuggingFace if needed)
- GPU detection and tensor parallelism initialization
- Server startup on port 30000
- Any errors during initialization

---

## Testing & Validation

### 6. Find Your Node

```bash
# Get the node where your job is running
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
```

**Intent**: Identify which compute node is running your SGLang server.
**Tests**: Nothing - informational query.

**Example output**:
```
JOBID PARTITION  NAME      USER  ST   TIME  TIME_LIMI  NODES NODELIST
12345 gpu        sglang-ll user  R    5:32  24:00:00   1     gpu-node-04
```

---

### 7. Health Check

```bash
# From login node, check if server is responding
# Replace gpu-node-04 with your actual node name
curl http://gpu-node-04:30000/health
```

**Intent**: Quick check that the server is running and responding.
**Tests**:
- Network connectivity to compute node
- Server is listening on port 30000
- Server process is alive

**Expected output**: `{"status": "ok"}` or similar health response.

---

### 8. Interactive Chat Test

```bash
# Start interactive chat session
make chat HOST=gpu-node-04 PORT=30000
```

**Intent**: Test end-to-end inference with conversational interface.
**Tests**:
- Server API is functional
- Model can generate responses
- **Inference speed** - displays tokens/second metrics
- Multi-turn conversation handling
- Server handles concurrent requests

**What you'll see**:
- Welcome banner with server info
- Prompt to enter messages
- Response from the model
- **Metrics**: Time elapsed, tokens/sec, total tokens

**Example interaction**:
```
You: What is tensor parallelism?
Assistant: [model response...]

[Metrics]
  Time: 2.45s
  Speed: 104.5 tokens/s
  Tokens: 256 total
```

**Commands during chat**:
- `/reset` - Clear conversation history
- `/quit` - Exit chat
- `/help` - Show help

---

### 9. API Test (Alternative)

```bash
# Test with curl (OpenAI-compatible API)
curl http://gpu-node-04:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3-70B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

**Intent**: Test the OpenAI-compatible API endpoint directly.
**Tests**:
- API endpoint functionality
- JSON request/response handling
- Single inference without chat context

---

## Monitoring

### 10. Check GPU Utilization (While Job Running)

```bash
# SSH to the compute node
ssh gpu-node-04

# Check GPU usage
nvidia-smi

# Check SGLang process
ps aux | grep sglang
```

**Intent**: Verify GPUs are being used efficiently.
**Tests**:
- All GPUs allocated to job are in use
- Memory utilization (~70-80% for 70B model)
- GPU compute utilization
- Tensor parallelism is active across GPUs

---

### 11. View Full Job Logs

```bash
# Check stdout
cat logs/sglang-JOBID.out

# Check stderr (if errors occurred)
cat logs/sglang-JOBID.err
```

**Intent**: Debug issues or review complete startup logs.
**Tests**: Full diagnostics - model loading, initialization, any errors.

---

## Cleanup

### 12. Cancel Job (When Done)

```bash
# Cancel the running job
scancel JOBID
```

**Intent**: Free up GPU resources when finished testing.
**Tests**: Job termination and resource cleanup.

---

## Troubleshooting Commands

### Job Won't Start

```bash
# Check job details
scontrol show job JOBID

# Check partition availability
sinfo -p gpu

# Check GPU availability
sinfo -o "%P %.5a %.10l %.6D %.6t %N %G"
```

**Tests**: Cluster resource availability, partition configuration.

---

### Server Not Responding

```bash
# Check if server process is running
ssh gpu-node-04 "ps aux | grep sglang"

# Check if port is listening
ssh gpu-node-04 "netstat -tuln | grep 30000"

# Check recent logs
ssh gpu-node-04 "tail -50 /path/to/logs/sglang-JOBID.out"
```

**Tests**: Process status, network listeners, recent errors.

---

### Out of Memory Errors

In `.env` or Slurm script, adjust:
```bash
# Reduce KV cache memory
MEM_FRACTION=0.80  # default is 0.85

# Or increase tensor parallelism
TENSOR_PARALLEL=8  # use more GPUs
```

**Tests**: Memory allocation strategies.

---

## Quick Reference

| Command | What It Tests |
|---------|---------------|
| `make validate` | Environment config, HF token, model access |
| `sbatch slurm/...` | Slurm job submission, resource availability |
| `squeue -u $USER` | Job status in queue |
| `curl .../health` | Server is alive and responding |
| `make chat` | **End-to-end inference speed and functionality** |
| `nvidia-smi` | GPU utilization and memory usage |

---

## Expected Timeline

1. **Job submission**: Immediate
2. **Queue wait**: 0-30 minutes (depends on cluster load)
3. **Model loading**: 5-15 minutes (first time, downloads ~140GB)
4. **Server ready**: ~2 minutes after model loaded
5. **First inference**: 2-10 seconds (depends on prompt length)
6. **Subsequent inferences**: ~2-5 seconds for 256 tokens

---

## Success Criteria

✅ Job starts and stays running
✅ All requested GPUs are allocated
✅ Model loads without OOM errors
✅ Server responds to `/health` endpoint
✅ Chat client connects successfully
✅ **Inference speed > 50 tokens/sec** (baseline for 70B on H100s)
✅ GPU utilization > 70%
✅ Multi-turn conversations work

---

## Key Metrics to Watch

When testing with `make chat`, you'll see metrics like:
```
[Metrics]
  Time: 2.45s
  Speed: 104.5 tokens/s    # THIS IS YOUR KEY METRIC
  Tokens: 256 total
```

**Typical performance for Llama 3 70B**:
- **4x H100 (TP=4)**: 80-120 tokens/sec
- **8x H100 (TP=8)**: 120-180 tokens/sec
- Lower = may indicate issues (network, memory, inefficient parallelism)
- Higher = excellent performance
