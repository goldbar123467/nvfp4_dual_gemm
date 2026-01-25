# SHARK TANK ROUND 2: LESSONS FROM THE WRECKAGE

---

## ROUND 1 RECAP: THE PIPELINE DISASTER

### What Happened

Round 1's unanimous winner **Pipeline Stages** (`num_ab_stage = 3`) was implemented and tested.

**RESULT: REGRESSION**

| Benchmark | Before (1 stage) | After (3 stages) | Change |
|-----------|------------------|------------------|--------|
| g=8, K=7168 | 373 µs | 488 µs | **-31% SLOWER** |
| g=8, K=2048 | 372 µs | 462 µs | **-24% SLOWER** |
| g=2, K=4096 | 173 µs | 249 µs | **-44% SLOWER** |
| g=2, K=1536 | 156 µs | 228 µs | **-46% SLOWER** |

### Why Pipeline Stages Failed

The sharks' analysis was based on standard GEMM assumptions that **don't apply to this kernel**:

1. **Already Memory-Efficient**: NVFP4 (4-bit) has extremely high compute-to-memory ratio. The tiny data size means TMA loads are already fast.

2. **Register Pressure**: Adding more stages increased register usage, possibly causing spilling or reducing occupancy.

3. **Shared Memory Overhead**: More stages = more SMEM = less occupancy = worse performance on small problems.

4. **Problem Size Mismatch**: Our small-M problems (40-248) may not have enough work to fill multiple pipeline stages efficiently.

5. **Dual GEMM Complexity**: The fused dual GEMM with SiLU may have different pipelining characteristics than standard single GEMM.

### The Lesson

**"Industry standard" optimizations don't always work.** This kernel is unique:
- NVFP4 format (4-bit, novel)
- Dual GEMM fusion
- Small M dimensions
- Block scaling factors

---

## ROUND 2 RULES

### For Contestants

You may:
1. **Repitch the same idea** with updated analysis explaining why Round 1 failed and how your approach differs
2. **Pitch a new idea** that accounts for the Round 1 learnings
3. **Modify your original pitch** based on the new data

**CRITICAL**: You must explain why your optimization won't suffer the same fate as Pipeline Stages.

### For Sharks

You must:
1. **Remember Round 1** - Pipeline Stages failed despite unanimous support
2. **Be more skeptical** of "industry standard" claims
3. **Demand kernel-specific evidence** not generic GEMM research
4. **Consider the unique aspects** of this kernel (NVFP4, dual GEMM, small M, block scaling)

### Scoring Adjustments

Given Round 1's failure, sharks should weight:
- **Kernel-Specific Analysis**: +10% weight (does the pitch address THIS kernel's unique characteristics?)
- **Risk Assessment**: +5% weight (what could go wrong?)
- **Incremental Testing Path**: +5% weight (can we test this safely without full implementation?)

---

## CURRENT KERNEL STATE

### Configuration (Reverted to Baseline)
```python
mma_tiler_mnk = (128, 128, 256)
num_ab_stage = 1  # Reverted from 3
num_acc_stage = 1
threads_per_cta = 128
```

### Baseline Performance
```
g=8, K=7168: 373 µs (target: 18.8 µs) - 19.8x from target
g=8, K=2048: 372 µs (target: 10.7 µs) - 34.8x from target
g=2, K=4096: 173 µs (target: 2.4 µs) - 72.1x from target
g=2, K=1536: 156 µs (target: 1.5 µs) - 102.4x from target
```

### Key Observations

1. **We're 20-100x off target** - Need dramatic improvements, not 1.5x tweaks
2. **Small M values dominate** - M ranges from 40-384, always << N
3. **Wave quantization is severe** - With 128x128 tiles and M=64, we get very few CTAs
4. **NVFP4 is unique** - Can't assume standard FP16/BF16 GEMM optimizations apply

---

## WHAT WE NEED

To hit targets, we need **5-100x improvement**, not 1.5x. Consider:

1. **Fundamentally different tile strategies** for small-M problems
2. **Algorithmic changes** that reduce total work
3. **Hardware features** we're not using (2SM instructions, cluster-level parallelism)
4. **Memory access patterns** specific to NVFP4 and block scaling

---

## ROUND 2 CONTESTANTS

Same 4 optimization categories, but pitches must be updated:

1. **Pipeline Stages** - Must explain why it failed and propose a fix (maybe 2 stages? Different configuration?)
2. **Tile Size Tuning** - Now more relevant given small-M problem
3. **TMA Store Epilogue** - Still valid, but must address overall kernel bottleneck
4. **Warp Specialization** - Must address complexity concerns

---

*"The first step to wisdom is admitting when you're wrong. The sharks were wrong. Now let's get it right."*

