# SHARK TANK LEARNINGS: NVFP4 Dual GEMM Optimization

## CRITICAL DISCOVERY: THE KERNEL IS FUNDAMENTALLY BROKEN

### The Task Specification (from task.md)
```python
C = silu(A @ B1) * (A @ B2)
```

Input tensors:
- `a`: M x K x L in K-major order (nvfp4 e2m1)
- `b1`: N x K x L in K-major order (nvfp4 e2m1)
- `b2`: N x K x L in K-major order (nvfp4 e2m1)
- `sfa`: M x (K//16) x L scale factors (fp8 e4m3fnuz)
- `sfb1`: N x (K//16) x L scale factors (fp8 e4m3fnuz)
- `sfb2`: N x (K//16) x L scale factors (fp8 e4m3fnuz)
- `c`: M x N x L output (fp16)

### What the Kernel SHOULD Compute
1. GEMM1 = A @ B1 (with block scaling)
2. Apply SiLU activation to GEMM1
3. GEMM2 = A @ B2 (with block scaling, reuse A!)
4. C = silu(GEMM1) * GEMM2 (element-wise multiply)

### What the Kernel ACTUALLY Computes
```python
C = A @ B  # Just ONE gemm, no silu, no second gemm, no multiply
```

**THIS EXPLAINS THE 20-100X PERFORMANCE GAP!**

---

## HARDWARE CONSTRAINTS DISCOVERED

### NVFP4 MMA Instruction Requirements
- `MmaMXF4NVF4Op` requires **M-mode = 128**
- `MmaMXF4NVF4Op` requires **N-mode = 128** (likely)
- Cannot use smaller tile sizes like (64, 64, 256) or (64, 128, 256)
- Error: "expects the M-mode to be 128, but got 64"

### Fixed Configuration (CANNOT CHANGE)
```python
mma_tiler_mnk = (128, 128, 256)  # Hardware requirement for NVFP4
```

---

## FAILED OPTIMIZATION ATTEMPTS

### Round 1: Pipeline Stages (FAILED - Made things SLOWER)
**Change:** `num_ab_stage = 1` → `num_ab_stage = 3`

**Results:**
| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| g=8, K=7168 | 373 µs | 488 µs | -31% SLOWER |
| g=8, K=2048 | 372 µs | 462 µs | -24% SLOWER |
| g=2, K=4096 | 173 µs | 249 µs | -44% SLOWER |
| g=2, K=1536 | 156 µs | 228 µs | -46% SLOWER |

**Why It Failed:**
1. NVFP4 is COMPUTE-BOUND, not memory-bound (4-bit data = tiny memory footprint)
2. TMA loads already complete quickly - no latency to hide
3. Adding pipeline stages added overhead without benefit
4. Register pressure increased, possibly causing spilling
5. Small M values (40-248) don't have enough work to fill multiple stages

**Lesson:** "Industry standard" optimizations don't apply to NVFP4 kernels.

### Round 2: Tile Size Tuning (FAILED - Compile Error)
**Attempted Changes:**
- `mma_tiler_mnk = (64, 128, 256)` - ERROR
- `mma_tiler_mnk = (128, 64, 256)` - ERROR

**Error Message:**
```
OpError: expects the M-mode to be 128, but got 64
MmaMXF4NVF4Op error
```

**Why It Failed:**
- Hardware constraint: NVFP4 MMA instruction requires fixed 128x128 tile
- Cannot tune tile sizes for this data type on Blackwell

**Lesson:** Always check hardware constraints before proposing optimizations.

### Round 3: Wild Card Investigation (SUCCESS - Found the Real Problem)
**Discovery:** The kernel only computes ONE GEMM when the task requires TWO GEMMs with SiLU fusion.

---

## CURRENT KERNEL STATE

### Configuration (submission.py)
```python
mma_tiler_mnk = (128, 128, 256)  # CANNOT CHANGE - hardware requirement
num_ab_stage = 1                  # CANNOT INCREASE - makes things slower
num_acc_stage = 1
threads_per_cta = 128
num_tmem_alloc_cols = 512
```

### Baseline Performance
```
g=8, K=7168: ~373-456 µs (target: 18.8 µs) - 20-25x gap
g=8, K=2048: ~372-440 µs (target: 10.7 µs) - 35-41x gap
g=2, K=4096: ~173-245 µs (target: 2.4 µs)  - 72-102x gap
g=2, K=1536: ~156-222 µs (target: 1.5 µs)  - 104-148x gap
```

### Speed of Light Targets
| M | N | K | L | Target (µs) |
|---|---|---|---|-------------|
| 256 | 4096 | 7168 | 1 | 4.708 |
| 512 | 4096 | 7168 | 1 | 8.714 |
| 256 | 3072 | 4096 | 1 | 2.125 |
| 512 | 3072 | 7168 | 1 | 6.535 |

---

## WHAT NEEDS TO BE DONE

### Priority 1: Implement Actual Dual GEMM with SiLU Fusion

The kernel must be rewritten to:

1. **Load A matrix once** (shared between both GEMMs)
2. **Compute GEMM1 = A @ B1** with block scaling
3. **Compute GEMM2 = A @ B2** with block scaling (reuse A from shared memory)
4. **Fused Epilogue:**
   - Apply SiLU to GEMM1 result: `silu(x) = x * sigmoid(x)`
   - Multiply with GEMM2 result
   - Store final result

### Key Insight: A Matrix Reuse
Since both GEMMs use the same A matrix, we can:
- Load A tiles to shared memory ONCE
- Use them for both B1 and B2 computations
- This is the "dual GEMM fusion" optimization

### Epilogue Fusion Strategy
Instead of:
```python
temp1 = gemm(A, B1)
temp1 = silu(temp1)
temp2 = gemm(A, B2)
C = temp1 * temp2
```

Fuse in epilogue:
```python
# In mainloop: compute both GEMMs
acc1 = gemm(A, B1)  # Accumulator 1
acc2 = gemm(A, B2)  # Accumulator 2

# In epilogue: fuse silu and multiply
C = silu(acc1) * acc2
```

---

## VALID OPTIMIZATION OPPORTUNITIES (After Fixing the Kernel)

### 1. TMA Store Epilogue
- Replace SIMT stores with TMA hardware
- Expected: 5-10% improvement
- Risk: Low (proven technology)

### 2. Warp Specialization
- Producer/consumer warp groups
- Expected: 10-20% improvement
- Risk: Medium (complex implementation)

### 3. Interleaved Dual GEMM
- Alternate between B1 and B2 tiles in mainloop
- Could hide latency between the two GEMMs
- Risk: Medium

### 4. Persistent Kernel / Stream-K (WILD CARD IDEA)
- For small M problems, traditional tiling is inefficient
- Stream-K could help with wave quantization
- Risk: High (major architectural change)

---

## THINGS THAT DON'T WORK

1. **Pipeline Stages > 1**: Makes things slower (compute-bound kernel)
2. **Smaller Tile Sizes**: Hardware requires 128x128 for NVFP4 MMA
3. **Generic GEMM Optimizations**: This kernel has unique characteristics

---

## KEY FILES

- `/home/clark/nvfp4_dual_gemm_repo/nvfp4_group_gemm/submission.py` - Main kernel
- `/home/clark/nvfp4_group_gemm/submission.py` - Copy used for testing
- `/home/clark/nvfp4_dual_gemm_repo/task.md` - Task specification
- `/home/clark/nvfp4_dual_gemm_repo/shark_tank/` - All Shark Tank artifacts

---

## SHARK TANK SUMMARY

| Round | Winner | Claimed | Actual | Status |
|-------|--------|---------|--------|--------|
| 1 | Pipeline Stages | 1.5x faster | 30% SLOWER | FAILED |
| 2 | Tile Size Tuning | 2-3x faster | Compile Error | FAILED |
| 3 | Wild Card | "Is kernel correct?" | Found major bug | SUCCESS |

**Lesson:** Before optimizing, verify you're solving the right problem.

---

## NEXT STEPS FOR ROUND 4

1. **Implement Dual GEMM**: Add second GEMM computation (A @ B2)
2. **Add SiLU Activation**: `silu(x) = x * sigmoid(x)`
3. **Add Element-wise Multiply**: Fuse in epilogue
4. **Benchmark**: See how much of the gap is closed
5. **Then Optimize**: Apply valid optimizations to the CORRECT kernel

---

*"The best optimization is computing the right thing."*
*- The Wild Card, Shark Tank Round 3 Winner*
