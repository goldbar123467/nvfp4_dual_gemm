# RAG Brain Update: NVFP4 Dual GEMM Optimization Learnings

**Date:** 2026-01-24
**Project:** NVFP4 Block-Scaled Group GEMM for gpumode.com
**GPU:** NVIDIA B200 (Blackwell)

---

## Key Discovery: 75% Thread Idle Problem

### The Problem
```python
# submission.py lines 316-354
if warp_idx == 0:  # ONLY 25% OF THREADS WORK!
    for k_tile in range(k_tile_cnt):
        # TMA loads
        # S2T copies
        # MMA operations ← ALL COMPUTE HERE
```

**128 threads available, only 32 used. 75% idle.**

### Root Cause Analysis
| Factor | Impact | Explanation |
|--------|--------|-------------|
| 75% threads idle | 4x | Only warp 0 works |
| No load/compute overlap | 1.5-2x | Single-stage pipeline |
| SIMT vs TMA stores | 1.2-1.5x | Inefficient epilogue |
| Two separate GEMMs | 1.3x | A matrix loaded twice |

---

## CuTe DSL Critical Constraints

### 1. Variable Scoping in Dynamic Control Flow
**ERROR:** `NameError: name 'acc_empty' is not defined`

**Cause:** Variables defined inside `if` blocks cannot be used in other `if` blocks.

**BAD:**
```python
if warp_idx == 0:
    acc_empty = acc_producer.acquire_and_advance()  # Defined here

# ... other code ...

if warp_idx == 0:
    acc_empty.commit()  # ERROR: acc_empty not in scope!
```

**GOOD:**
```python
# Define outside control flow
acc_empty = acc_producer.acquire_and_advance()

# ... other code ...

acc_empty.commit()  # Works: variable in scope
```

**DSL Suggestion:** "Using variables defined in dynamic control flow is not supported. Please give an initial value before control flow."

### 2. cute.gemm() is Cooperative
- `cute.gemm()` requires ALL threads in the participating group to execute it
- Each thread has its slice via `thr_mma = tiled_mma.get_slice(tidx)`
- Having only warp 0 execute means 75% of work doesn't happen

### 3. TMA Loads are Hardware-Constrained
- TMA (Tensor Memory Accelerator) loads must be issued by a single warp
- This is a Blackwell hardware requirement, not a software choice
- Cannot parallelize TMA across warps

### 4. S2T Copies are Thread-Constrained
- TMEM (Tensor Memory) writes require single-thread coordination
- Scale factor copies to TMEM must be done by thread 0

---

## Pipeline Configuration Patterns

### Producer/Consumer Groups
```python
# Producer = who issues loads/acquires
# Consumer = who waits/computes/releases

ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)  # Single thread
ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, threads_per_cta)  # All 128
```

### Double Buffering
```python
num_ab_stage = 2  # Enables load/compute overlap
```

**Without double buffering:** Load → Wait → Compute → Load → Wait → Compute
**With double buffering:** Load[0] → [Compute[0] + Load[1]] → [Compute[1] + Load[2]] → ...

---

## Optimization Attempts and Results

### Round 8A: All-Warps MMA
**Hypothesis:** Moving MMA outside `if warp_idx == 0` will give 4x speedup.

**Changes:**
- Consumer group: 1 thread → 128 threads
- MMA execution: Warp 0 only → All threads

**Result:** Only 3% improvement (466µs vs 479µs)

**Learning:** The cooperative MMA might already be handled internally by the DSL, or there's another bottleneck.

### Round 8B: Double Buffering + Pipelined Loads
**Hypothesis:** Overlapping loads with compute will hide memory latency.

**Changes:**
- `num_ab_stage = 2` (double buffering)
- Prologue: Issue first load before loop
- Main loop: Issue load[N+1] while computing tile[N]

**Structure:**
```python
# Prologue
if warp_idx == 0:
    issue_load(tile_0)

# Main loop
for k_tile in range(k_tile_cnt):
    wait_for_tile(k_tile)

    if warp_idx == 0 and k_tile < k_tile_cnt - 1:
        issue_load(k_tile + 1)  # OVERLAP

    compute_mma(k_tile)  # ALL THREADS
    release(k_tile)
```

---

## Hardware Specifications (B200 Blackwell)

| Spec | Value |
|------|-------|
| SMs | 192 |
| FP4 Tensor Cores | Yes |
| Memory Bandwidth | ~2.25 TB/s |
| MMA Tile Size | 128x128 minimum (NVFP4) |
| Threads per CTA | 128 (4 warps) |
| Threads per Warp | 32 |

---

## Performance Targets

| M | N | K | L | Target (µs) | Current (µs) | Gap |
|---|---|---|---|-------------|--------------|-----|
| 256 | 4096 | 7168 | 1 | 4.7 | ~466 | 99x |
| 512 | 4096 | 7168 | 1 | 8.7 | - | - |

---

## Untried Optimizations

| Optimization | Expected Gain | Risk | Status |
|--------------|---------------|------|--------|
| True Dual-GEMM Fusion | 30%+ | High | Not tried |
| TMA Store Epilogue | 20-30% | Medium | Not tried |
| 128-bit Vector Stores | 10-20% | Low | Not tried |
| Persistent Kernels | 2x+ | High | Not tried |

---

## Code Patterns Learned

### Correct Main Loop Structure
```python
# Define pipeline tokens outside control flow
acc_empty = acc_producer.acquire_and_advance()
tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

# Prologue: Warp 0 issues first load
if warp_idx == 0:
    ab_empty = ab_producer.acquire_and_advance()
    cute.copy(tma_atom_a, ...)
    # ... other TMA copies

# Main loop
for k_tile in range(k_tile_cnt):
    # ALL threads wait
    ab_full = ab_consumer.wait_and_advance()

    # Warp 0 issues next load (overlap)
    if warp_idx == 0:
        if k_tile < k_tile_cnt - 1:
            # Issue next tile's load

    # Thread 0 does S2T
    if tidx == 0:
        cute.copy(tiled_copy_s2t_sfa, ...)

    # Barrier before MMA
    cute.arch.barrier()

    # ALL threads do MMA
    cute.gemm(tiled_mma, ...)

    ab_full.release()

# Commit outside control flow
acc_empty.commit()
```

---

## Shark Tank Methodology

**Format:** AI agents pitch optimizations, 3 "shark" judges vote on implementation.

**Scoring Criteria:**
- Technical feasibility
- Expected performance gain
- Implementation risk
- Code complexity

**Lesson:** Research before optimization. Round 7's research phase revealed the 75% idle thread problem that 6 rounds of blind optimization missed.

---

## Tags
`cuda` `cutlass` `cute-dsl` `nvfp4` `blackwell` `b200` `gemm` `optimization` `warp-specialization` `double-buffering` `tma` `mma` `pipeline`
