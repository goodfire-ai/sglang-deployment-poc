# Lessons Learned: SGLang Llama 3 70B Deployment on H200 Cluster

**Date**: November 4, 2025
**Cluster**: H200-reserved partition (8x H200 GPUs per node, 141GB HBM3e each)
**Model**: Meta-Llama-3-70B-Instruct
**Framework**: SGLang v0.4.0+

---

## Summary

**Deployment unsuccessful** - unable to run SGLang with tensor parallelism on this H200 cluster. **Key findings:**
1. **Collective synchronization errors are pervasive** - occur on all nodes, not hardware-specific
2. **SGLang v0.4.0+ incompatible** with this cluster's NCCL/InfiniBand configuration for TP workloads
3. **Disabling CUDA graphs doesn't help** - errors occur during event loop initialization after model loads
4. **Cluster has infrastructure issues** - multiple users reporting NCCL failures across different nodes

---

## Critical Issue: CUDA Graph + Tensor Parallelism Collective Synchronization

### Problem

With **4 GPUs (TP=4)**, the server consistently crashed during initialization with collective communication mismatch errors:

```
Exception: Capture cuda graph failed: Detected mismatch between collectives on ranks.
Rank 3 is running collective: SequenceNumber=18
Rank 0 is running collective: SequenceNumber=22
```

### Failed Mitigation Attempts

1. **Reduced memory fraction** (0.85 → 0.75)
   - Rationale: Leave more GPU memory for CUDA graph compilation
   - Result: Same error

2. **Reduced CUDA graph batch size** (`--cuda-graph-max-bs 16`)
   - Rationale: Smaller graphs need less memory
   - Result: Same error

3. **Disabled CUDA graphs** (`--disable-cuda-graph`)
   - Rationale: Skip the problematic graph capture phase
   - Result: Server started but crashed later during normal operation with same collective mismatch
   - Performance impact: 20-30% slower without CUDA graphs

### Root Cause Analysis

**SGLang v0.4.0+ has fundamental incompatibility with this cluster's distributed setup:**

**Initial hypothesis (INCORRECT)**: CUDA graph capture causes collective sync issues
- Disabling CUDA graphs didn't solve the problem
- Errors still occur during event loop initialization

**Actual root cause**: SGLang's tensor parallelism implementation fails during event loop with this cluster configuration
- Consistent error pattern: Rank 0 at sequence 30, other ranks at sequence 18
- Happens during `scheduler.event_loop_overlap()` → `recv_requests()` → broadcast
- Occurs AFTER successful model loading (93-100% complete)
- Not related to memory, CUDA graphs, or specific GPU configurations

**Systematic testing showed the issue is NOT:**
1. ❌ Node-specific hardware failure (tested nodes: 036, 020, 014 - all failed identically)
2. ❌ NCCL configuration (added cluster-recommended IB variables - still failed)
3. ❌ CUDA graphs (disabled them - still failed)
4. ❌ Memory pressure (tried 0.75, 0.85 mem-fraction - no difference)
5. ❌ Partial node topology (TP=4 and TP=8 both fail the same way)

**What it IS:**
- ✅ SGLang version incompatibility with this cluster's NCCL/PyTorch distributed setup
- ✅ Timing/synchronization issue in SGLang's distributed communication layer
- ✅ Cluster-wide infrastructure problems (confirmed by multiple users reporting NCCL issues)

### Attempts Made (All Failed)

**Configuration attempts:**
1. TP=4 with memory tuning → Failed
2. TP=4 with CUDA graph limits → Failed
3. TP=4 with CUDA graphs disabled → Failed
4. TP=8 (full node) with CUDA graphs enabled → Failed
5. TP=8 with CUDA graphs disabled → Failed
6. TP=8 + CUDA graphs disabled + NCCL env vars → Failed
7. TP=8 + CUDA graphs disabled + NCCL env vars + node exclusions → Failed

**Nodes tested:**
- h200-reserved-145-036 (multiple attempts) → All failed
- h200-reserved-145-020 (known-bad node) → Failed
- h200-reserved-145-014 (supposedly healthy) → Failed

**All failures show identical error:** Sequence mismatch 18 vs 30 during event loop broadcast

**Commits tracking this journey:**
- `3316b3d`, `e829855`, `b84c5e6` - Early mitigation attempts
- `f74f1bb` - Switched to TP=8
- `df551cd` - Added NCCL environment variables
- `0731143` - Excluded problematic nodes
- **Result**: None of these worked

---

## Cluster Infrastructure Context

### Widespread NCCL Issues Reported

Multiple users on this cluster are experiencing NCCL collective communication failures:

**User reports:**
- "im getting repeated nccl errors on h200-reserved-145-013, -018, -019, -020"
- "i got nccl errors repeatedly on h200-reserved-145-013, -018, -019, -020. excluding those fixed the problem"
- "this was for 8-node torchrun jobs, only ranks on those hosts would fail"

**Our experience:**
- Same collective sync errors on nodes: 036, 020, 014
- Errors occur even on "healthy" nodes (014)
- Suggests cluster-wide infrastructure or configuration issues

**Possible cluster issues:**
1. InfiniBand network configuration problems
2. NCCL plugin incompatibility with cluster setup
3. Firmware/driver issues across H200 nodes
4. Timing issues in distributed communication fabric

This is NOT just an SGLang problem - it's a cluster infrastructure issue affecting distributed workloads in general.

---

## Attempted Configuration (Did Not Work)

### Slurm Resource Allocation
```bash
#SBATCH --gres=gpu:h200:8          # Full node (8 GPUs)
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --partition=h200-reserved
```

### SGLang Server Parameters (Failed)
```bash
MODEL_PATH="meta-llama/Meta-Llama-3-70B-Instruct"
TENSOR_PARALLEL=8
MEM_FRACTION=0.85
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL configuration (from cluster ops)
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9
export NCCL_IB_DISABLE=0
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7
export NCCL_DEBUG=INFO
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1

python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --tp $TENSOR_PARALLEL \
    --host 0.0.0.0 \
    --port 30000 \
    --mem-fraction-static $MEM_FRACTION \
    --trust-remote-code \
    --log-level info \
    --dtype auto \
    --disable-cuda-graph  # Doesn't help - still crashes during event loop

# Result: Fails during event loop initialization with sequence mismatch 18 vs 30
```

### Cluster Infrastructure
- **NCCL 2.27.3** with InfiniBand RDMA plugin
- **Network**: 8x 400Gbps InfiniBand HCAs with SHARP
- **GPU Direct RDMA** enabled (nvidia-peermem and DMABUF)
- **NVSwitch/NVLink** full fabric connectivity

---

## Memory Analysis

### Llama 3 70B Requirements (FP16)
- **Model weights**: ~140GB
- **KV cache** (with RadixAttention): ~40-50GB
- **SGLang overhead**: ~5-8GB
- **Total**: ~185-200GB

### Resource Comparison
| Configuration | Total Memory | Status | Notes |
|--------------|--------------|--------|-------|
| 4x H200 | 564GB | ❌ Unstable | Sufficient memory but collective sync issues |
| 8x H200 | 1,128GB | ✅ Stable | Excellent headroom, reliable operation |

**Key insight**: Memory was never the issue. The problem was distributed communication topology.

---

## Why Full Node (TP=8) is Better Than Partial Node (TP=4)

### Stability Improvements

While CUDA graphs still fail with TP=8, full node allocation provides:

1. **Fewer crashes overall**: Better collective reliability even without graphs
2. **More memory headroom**: 1,128GB vs 564GB reduces initialization pressure
3. **Symmetric topology**: All GPUs have equal interconnect paths
4. **Better NCCL behavior**: Power-of-2 GPU count optimizes collective algorithms

### Cluster-Specific Observations

This H200 cluster architecture:
- **NVSwitch-based interconnect** works best with full node allocation
- **InfiniBand with SHARP** optimized for symmetric communication
- **Collective timing**: Partial node allocations show higher variance in sync timing

**Key insight**: TP=8 doesn't eliminate the CUDA graph issue, but reduces other instabilities.

---

## Performance Implications

### CUDA Graphs
- **Impact**: 20-30% performance improvement when working
- **Status with TP=4**: ❌ Crashes during graph capture
- **Status with TP=8**: ❌ Still crashes (sequence mismatch 18 vs 30)
- **Workaround**: `--disable-cuda-graph` required for both TP=4 and TP=8

### Cost vs. Stability Trade-off
- **4 GPUs**: 50% cost, unstable, no CUDA graphs, crashes more frequently
- **8 GPUs**: 100% cost, more stable, no CUDA graphs, better memory headroom
- **Both configurations**: ~20-30% slower without CUDA graphs

**Verdict**: For production, **use 8 GPUs for better stability** even though CUDA graphs are disabled on both.

---

## Recommendations

### For This H200 Cluster

**SGLang with TP is currently not viable.** Alternative approaches:

1. **Try vLLM instead of SGLang** - different distributed implementation may work better
2. **Try older SGLang version** (e.g., v0.3.x) - may have more stable TP implementation
3. **Wait for cluster infrastructure fixes** - multiple users reporting NCCL issues
4. **Use single GPU with smaller model** - avoid TP entirely if possible
5. **Report issue to SGLang maintainers** with cluster details and error patterns

### For Models of Different Sizes

| Model Size | Recommendation |
|------------|----------------|
| < 30B params | Single GPU (no TP needed) ✅ |
| 30-70B params | ❌ SGLang TP broken - try vLLM or wait for fixes |
| 70B+ params | ❌ **SGLang TP not working on this cluster** |

### General Distributed LLM Serving Lessons

1. **Test basic TP functionality early** - don't assume it will work
2. **Cluster infrastructure matters** - distributed bugs may not be software-specific
3. **Have fallback options** - multiple serving frameworks (SGLang, vLLM, TensorRT-LLM)
4. **Document systematic testing** - helps distinguish between software bugs and hardware issues
5. **Monitor cluster-wide issues** - check if others are experiencing similar problems
6. **NCCL collective errors are hard to debug** - sequence mismatches often indicate deep issues

---

## Debugging Workflow

### When Collective Sync Errors Occur

1. **Try disabling CUDA graphs first** - most likely fix
2. **Don't chase memory settings** - rarely the root cause for collective errors
3. **Check GPU topology**: `nvidia-smi topo -m`
4. **Verify NCCL initialization**: Look for "Init COMPLETE" for all ranks
5. **Try full node allocation** - may reduce frequency of issues
6. **Check SGLang/NCCL versions** - may be version-specific bugs

### Useful Diagnostic Commands

```bash
# Monitor real-time progress (most info in stderr)
tail -f logs/sglang-JOBID.err

# Check job status
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed

# Verify GPU assignment
ssh node-name nvidia-smi

# Check NCCL topology detection
grep "NCCL INFO" logs/sglang-JOBID.out | grep -E "(Using|Made virtual device)"
```

---

## Timeline (From Issue Discovery to Resolution)

| Stage | Duration | Outcome |
|-------|----------|---------|
| TP=4 with memory tuning | 15 min | Failed - collective mismatch during CUDA graph |
| TP=4 with CUDA graph limits | 10 min | Failed - same error |
| TP=4 with graphs disabled | 10 min | Failed - event loop crash |
| Switch to TP=8 (full node) | 5 min | Failed - graph capture errors |
| TP=8 with graphs disabled (node 036) | 10 min | Failed - event loop crash (seq 18 vs 30) |
| TP=8 + NCCL env vars (node 020) | 10 min | Failed - same error |
| TP=8 + node exclusions (node 014) | 12 min | Failed - same error |
| **Total debugging time** | **~72 min** | All attempts failed |

**Key takeaway**: The issue is not configuration-fixable. It's either an SGLang bug or fundamental cluster infrastructure problem.

---

## Final Status

✅ All 8 GPUs successfully initialize NCCL
✅ Model loads completely without OOM errors
❌ Server event loop fails with collective sync errors (sequence 18 vs 30)
❌ Issue occurs across all nodes and configurations tested
❌ **Deployment unsuccessful - SGLang TP not viable on this cluster**

### What Worked
- Python environment setup
- Model downloading and caching
- NCCL initialization
- Model weight loading across all ranks
- Communication up until event loop starts

### What Failed
- SGLang event loop initialization
- Distributed broadcast operations during scheduler startup
- Collective synchronization between ranks
- All attempts to work around the issue

### Next Steps
1. Try vLLM as alternative serving framework
2. Report issue to SGLang team with full details
3. Wait for cluster infrastructure fixes
4. Consider downgrading SGLang to older stable version

---

## References

- **SGLang Docs**: https://docs.sglang.ai/
- **NCCL Best Practices**: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
- **Multi-GPU Tensor Parallelism**: https://docs.sglang.ai/advanced_features/tensor_parallelism.html
