# Gap 6: Nsight Compute Proof of Tensor Core Path

## Overview

It is not enough to simply "profile" a kernel. You must confirm:
1. Tensor core instructions actually execute (WGMMA / tcgen05 path on SM100)
2. No register spills limiting performance
3. DRAM throughput vs compute aligns with roofline expectations

This document provides the verification methodology for proving tensor core utilization.

## Required Evidence

### 1. Tensor Core Instructions Present

```bash
# Profile command
ncu --set full \
    --metrics sm__pipe_tensor_op_hmma_cycles_active,\
              sm__inst_executed_pipe_tensor_op_hmma.sum,\
              sm__pipe_tensor_op_hmma_utilization \
    ./nvfp4_dual_gemm --m=512 --n=4096 --k=7168

# For Blackwell SM100 specifically, look for:
# - tcgen05.mma instructions (FP4 tensor ops)
# - WGMMA (Warp Group MMA) utilization
```

**What to look for:**
- `sm__pipe_tensor_op_hmma_cycles_active` > 0
- `sm__inst_executed_pipe_tensor_op_hmma.sum` shows tensor instructions
- NOT seeing only CUDA core FMA instructions

**Critical SM100/Blackwell metrics:**
```bash
# Blackwell-specific tensor core metrics
ncu --metrics \
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
    smsp__warp_issue_stalled_mio_throttle.avg.pct_of_peak_sustained_elapsed,\
    smsp__inst_executed_pipe_tensor_op_hmma.sum \
    ./nvfp4_dual_gemm
```

### 2. No Register Spills

```bash
ncu --metrics \
    launch__registers_per_thread,\
    lts__t_sectors_srcunit_local_op_ld.sum,\
    lts__t_sectors_srcunit_local_op_st.sum \
    ./nvfp4_dual_gemm --m=512 --n=4096 --k=7168
```

**What to look for:**
- `launch__registers_per_thread` < 255 (max per thread)
- `lts__t_sectors_srcunit_local_op_ld.sum` approximately 0 (no local memory loads)
- `lts__t_sectors_srcunit_local_op_st.sum` approximately 0 (no local memory stores)

**If spilling detected:**
- Reduce tile size
- Reduce register usage in epilogue
- Check for unnecessary intermediates
- Consider `__launch_bounds__` to constrain register usage

### 3. Memory vs Compute Alignment

```bash
ncu --metrics \
    dram__bytes.sum,\
    dram__throughput.avg.pct_of_peak_sustained_elapsed,\
    sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active,\
    gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed \
    ./nvfp4_dual_gemm --m=512 --n=4096 --k=7168
```

**Roofline interpretation:**
```
B200 specs:
- FP4 Tensor Core peak: ~18 PFLOPS
- HBM bandwidth: ~8 TB/s
- Ridge point: 18000 TFLOP/s / 8 TB/s = 2250 ops/byte

For problem (512, 4096, 7168):
# CORRECTED CALCULATION (previous version had 1000x error!)
# Formula: FLOPs = 2*GEMM1 + 2*GEMM2 + SiLU(3*M*N) + Mul(M*N)
#        = 4*M*N*K + 4*M*N

- FLOPs = 4 * 512 * 4096 * 7168 + 4 * 512 * 4096
        = 60,129,542,144 + 8,388,608
        = 60.14 GFLOP (NOT TFLOP!)

# Memory breakdown:
- A:     512 * 7168 / 2     =  1,835,008 bytes (FP4 packed)
- B1:  4,096 * 7168 / 2     = 14,680,064 bytes (FP4 packed)
- B2:  4,096 * 7168 / 2     = 14,680,064 bytes (FP4 packed)
- sfa:   512 * 448          =    229,376 bytes (FP8, K/16 scale blocks)
- sfb1: 4096 * 448          =  1,835,008 bytes (FP8)
- sfb2: 4096 * 448          =  1,835,008 bytes (FP8)
- C:     512 * 4096 * 2     =  4,194,304 bytes (FP16 output)
- Total:                    = 39,288,832 bytes = ~39.3 MB

- Arithmetic Intensity = 60.14 GFLOP / 39.3 MB = ~1.53 ops/byte

# Analysis:
AI = 1.53 ops/byte << Ridge point (2250 ops/byte)
=> STRONGLY MEMORY BOUND as expected for inference workloads
- Should see high DRAM throughput (>80%)
- Tensor core utilization will be limited by memory bandwidth
- Speed of Light target (8.714 us) is memory-bound target
```

### 4. Occupancy and Shared Memory

```bash
ncu --metrics \
    sm__warps_active.avg.pct_of_peak_sustained_active,\
    sm__maximum_warps_per_active_cycle_pct,\
    shared_efficiency,\
    l1tex__m_xbar2l1tex_read_bytes.sum \
    ./nvfp4_dual_gemm --m=512 --n=4096 --k=7168
```

**What to look for:**
- Occupancy > 50% (ideally >75%)
- Shared memory bank conflicts low
- L1 cache hit rate reasonable

## Complete Profiling Command

```bash
# Full report with all critical metrics
ncu --set full \
    --section ComputeWorkloadAnalysis \
    --section MemoryWorkloadAnalysis \
    --section SpeedOfLight \
    --section Roofline \
    --export nvfp4_profile_%i \
    --target-processes all \
    ./nvfp4_dual_gemm --m=512 --n=4096 --k=7168

# Generate roofline chart data
ncu --page details --csv nvfp4_profile_0.ncu-rep > roofline_data.csv
```

## Validation Gate: Screenshot Checklist

In your Nsight Compute report, you must be able to point to:

### Tensor Instruction Utilization
- [ ] Screenshot showing tensor pipe active cycles
- [ ] Confirm WGMMA/tcgen05 instructions (not FMA fallback)

### Achieved Performance
- [ ] FLOP/s or tensor pipe throughput
- [ ] Compare to theoretical peak

### Memory Throughput
- [ ] DRAM bytes transferred
- [ ] Throughput % of peak
- [ ] Verify matches expected bytes from roofline calc

### Occupancy and Resources
- [ ] Warps per SM
- [ ] Registers per thread
- [ ] Shared memory per block
- [ ] No local memory spills

## Example Good vs Bad Report

### Good (Tensor Cores Working):
```
sm__pipe_tensor_op_hmma_cycles_active: 85.2%
dram__throughput.pct_of_peak: 78.3%
launch__registers_per_thread: 128
local memory spills: 0
```

### Bad (Fallback to CUDA Cores):
```
sm__pipe_tensor_op_hmma_cycles_active: 0%
sm__inst_executed_pipe_fma.sum: 10000000  <- CUDA core fallback!
```

### Bad (Register Spilling):
```
lts__t_sectors_srcunit_local_op_st.sum: 500000  <- Spilling!
```

## Automation Script

```bash
#!/bin/bash
# verify_tensor_core.sh - Automated tensor core verification

BINARY="./nvfp4_dual_gemm"
ARGS="--m=512 --n=4096 --k=7168"

echo "=== Tensor Core Verification ==="

# Check tensor instructions
echo "1. Checking tensor core instructions..."
ncu --metrics sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active \
    $BINARY $ARGS 2>&1 | grep -E "hmma_cycles_active|tensor"

# Check for register spills
echo "2. Checking for register spills..."
ncu --metrics lts__t_sectors_srcunit_local_op_ld.sum,lts__t_sectors_srcunit_local_op_st.sum \
    $BINARY $ARGS 2>&1 | grep -E "local_op"

# Check memory throughput
echo "3. Checking memory throughput..."
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    $BINARY $ARGS 2>&1 | grep -E "dram__throughput"

echo "=== Verification Complete ==="
```

## Troubleshooting

### No Tensor Core Activity
1. Check CUTLASS version supports SM100/Blackwell
2. Verify FP4 MMA instruction exists for your GPU arch
3. Check alignment requirements (16-byte for most tensor ops)
4. Verify data types match tensor core requirements

### High Register Usage
1. Add `__launch_bounds__(threads, minBlocks)` directive
2. Reduce tile size in CuTe layout
3. Simplify epilogue computation
4. Check for unintended array allocations

### Memory Bound Despite Low AI
This is expected for inference workloads. Focus on:
1. Maximizing DRAM throughput
2. Minimizing kernel launch overhead
3. Fusing operations where possible
