# ARCHAEOLOGIST REPORT: NVFP4 Dual GEMM Shark Tank

## Executive Summary

This repository documents an AI-driven optimization competition for an NVFP4 Block-Scaled Dual GEMM kernel targeting NVIDIA B200 (Blackwell) GPUs. The project uses a "Shark Tank" format where AI agents pitch optimizations that are evaluated by other AI "shark" agents.

**Key Finding:** The kernel was fundamentally broken for the first 3 rounds - computing a single GEMM instead of the required dual-GEMM with SiLU fusion.

---

## 1. FILE INVENTORY

### Root Level Files

| Path | Purpose | Relevance | Notes |
|------|---------|-----------|-------|
| `/README.md` | Project overview with Shark Tank format explanation | CRITICAL | Documents methodology, season results, key learnings |
| `/task.md` | Official challenge specification from GPUMode | CRITICAL | Defines C = silu(A @ B1) * (A @ B2), speed-of-light targets |
| `/SHARK_TANK_LEARNINGS.md` | Accumulated wisdom from all rounds | CRITICAL | Documents all failures and successes, key constraints discovered |
| `/CLAUDE.md` | Agent coordination and project context | USEFUL | Multi-agent swarm configuration, project structure |
| `/SHARK1_SCORECARD.txt` | Round 8 Shark 1 detailed analysis | USEFUL | Deep technical insights on thread utilization |
| `/rag_brain_update.md` | RAG brain memory update notes | USEFUL | Context for knowledge storage |
| `/.gitignore` | Git ignore patterns | USEFUL | Standard configuration |

### Shark Tank Directory (`/shark_tank/`)

| Path | Purpose | Relevance | Notes |
|------|---------|-----------|-------|
| `/shark_tank/README.md` | Shark Tank format rules | CRITICAL | Defines scoring criteria, contestant structure |
| `/shark_tank/rounds/round1_results.md` | Round 1 Pipeline Stages results | CRITICAL | FAILED: 30% SLOWER |
| `/shark_tank/rounds/round2_results.md` | Round 2 Tile Size Tuning results | CRITICAL | FAILED: COMPILE ERROR |
| `/shark_tank/rounds/round3_results.md` | Round 3 Wild Card Debug results | CRITICAL | SUCCESS: Found the kernel was wrong |
| `/shark_tank/rounds/round4_results.md` | Round 4 Minimal Fix results | CRITICAL | SUCCESS: Two-pass implementation |
| `/shark_tank/rounds/round5_results.md` | Round 5 Stream Parallelism results | CRITICAL | BLOCKED: Competition rules forbid streams |
| `/shark_tank/rounds/round6_results.md` | Round 6 Reduce Overhead results | CRITICAL | FAILED: 33% SLOWER (pre-alloc didn't help) |
| `/shark_tank/rounds/round7_results.md` | Round 7 Research results | CRITICAL | SUCCESS: Found 75% threads idle |
| `/shark_tank/rounds/round8_results.md` | Round 8 Fix Warp Usage results | CRITICAL | READY TO TEST: All warps in MMA |
| `/shark_tank/rounds/round8_implementation.md` | Round 8 implementation details | CRITICAL | Code changes documented |
| `/shark_tank/rounds/*_context.md` | Round context files | USEFUL | Background for each round |
| `/shark_tank/rounds/*_shark*_vote.md` | Individual shark votes | USEFUL | Detailed reasoning from each shark |

### Shark Tank Pitches (`/shark_tank/pitches/`)

| Path | Purpose | Relevance | Notes |
|------|---------|-----------|-------|
| `contestant1_pipeline_stages.md` | Pipeline stages pitch | USEFUL | Round 1 winner (failed) |
| `contestant2_tile_sizes.md` | Tile size tuning pitch | USEFUL | Round 2 winner (failed) |
| `contestant3_tma_epilogue.md` | TMA epilogue pitch | USEFUL | Never implemented |
| `contestant4_warp_specialization.md` | Warp specialization pitch | USEFUL | Multiple rounds, eventually partial success |
| `round3_contestantC_wild_card.md` | Wild card debug pitch | CRITICAL | Found the real bug |
| `round4_contestantC_minimal.md` | Two-pass minimal fix | CRITICAL | Implemented successfully |
| `round5_wildcardA_triton.md` | Triton rewrite pitch | DEAD | Triton cannot access NVFP4 MMA |
| `round5_wildcardB_compile.md` | torch.compile pitch | DEAD | Unverified assumptions |
| `round5_wildcardC_streams.md` | Stream parallelism pitch | BLOCKED | Competition rules forbid |

### NVFP4 Group GEMM (`/nvfp4_group_gemm/`)

| Path | Purpose | Relevance | Notes |
|------|---------|-----------|-------|
| `submission.py` | Current best submission (v8b) | CRITICAL | Main kernel with Round 8 warp fix |
| `submission_v8_prealloc.py` | Round 6 pre-allocation attempt | USEFUL | Failed - made things 33% slower |
| `submission_v7_final.py` | Pre-Round 8 version | USEFUL | Baseline for warp fix comparison |
| `submission_v6_clean.py` | Earlier clean version | DEAD | Superseded |
| `RESEARCH.md` | PyTorch optimization research | CRITICAL | API docs, scale factor layouts, optimization strategies |

### Python Directory (`/python/`)

| Path | Purpose | Relevance | Notes |
|------|---------|-----------|-------|
| `task.py` | Task definition with input generation | CRITICAL | Reference implementation, validation |
| `submission.py` | Simple PyTorch submission | USEFUL | Basic approach |
| `submission_best*.py` | Various optimization attempts | EXPERIMENTAL | Multiple versions tried |
| `submission_streams.py` | Stream parallelism attempt | BLOCKED | Competition forbids streams |
| `submission_triton.py` | Triton kernel attempt | DEAD | Cannot access FP4 MMA |
| `submission_compile.py` | torch.compile attempt | EXPERIMENTAL | Uncertain benefits |
| `submission_prealloc.py` | Pre-allocation attempt | DEAD | Didn't help |
| `submission_fused.py` | Fused operations attempt | EXPERIMENTAL | |
| `utils.py` | Utility functions | USEFUL | `make_match_reference` validation |
| `constants.py` | Shared constants | USEFUL | |
| `kernel.py` | Kernel wrapper | USEFUL | |
| `benchmark_submission.py` | Benchmarking utilities | USEFUL | |
| `test_kernel.py` | Tests | USEFUL | |

### Documentation (`/docs/`)

| Path | Purpose | Relevance | Notes |
|------|---------|-----------|-------|
| `PITCH_REDUCE_LAUNCH_OVERHEAD.md` | Round 6 pitch document | CRITICAL | Detailed Python overhead analysis |
| `phase1_*.md` | Phase 1 setup/test reports | USEFUL | Initial analysis |
| `gap1_b_layout_transpose.md` | B matrix layout gap | USEFUL | Technical detail |
| `gap2_scale_factor_mapping.md` | Scale factor mapping gap | USEFUL | Technical detail |
| `gap3_fp4_packing.md` | FP4 packing gap | USEFUL | Technical detail |
| `gap4_precision_policy.md` | Precision policy gap | USEFUL | Technical detail |
| `gap5_benchmark_harness.md` | Benchmark harness gap | USEFUL | Technical detail |
| `gap6_nsight_proof.md` | Nsight profiling proof | USEFUL | Technical detail |
| `build.md` | Build instructions | USEFUL | |
| `build-guide/*.md` | Build guide subdocs | USEFUL | CMake reference, troubleshooting |

### Source Code (`/src/`)

| Path | Purpose | Relevance | Notes |
|------|---------|-----------|-------|
| `kernel/nvfp4_dual_gemm.cu` | CUDA kernel source | USEFUL | Alternative implementation |
| `nvfp4_dual_gemm.cpp` | C++ wrapper | USEFUL | PyBind interface |
| `silu_mul_kernel.cu` | SiLU multiplication kernel | USEFUL | Fused operation |
| `epilogue/left_silu_and_mul.h` | Epilogue header | USEFUL | Custom epilogue |
| `setup.py` | Build configuration | USEFUL | |
| `test_kernel.py` | Source tests | USEFUL | |

### Tests (`/tests/`)

| Path | Purpose | Relevance | Notes |
|------|---------|-----------|-------|
| `run_all_tests.py` | Test runner | USEFUL | |
| `test_step1_dtype_consistency.py` | Data type tests | USEFUL | |
| `test_step2_fp4_nibble_order.py` | FP4 packing tests | USEFUL | |
| `test_step3_flops_and_contiguity.py` | Performance tests | USEFUL | |

---

## 2. ROUND-BY-ROUND KNOWLEDGE EXTRACTION

### ROUND 1: Pipeline Stages

**Optimization Proposed:** Increase `num_ab_stage` from 1 to 3 for better memory latency hiding.

**Shark Votes:**
| Shark | Vote | Score | Reasoning |
|-------|------|-------|-----------|
| Performance Oracle | Pipeline Stages | 8.4/10 | "180-360% speedup per hour of work" |
| Pragmatic Engineer | Pipeline Stages | 8.05/10 | "The best optimization is the one you can ship this week" |
| ROI Maximizer | Pipeline Stages | 8.9/10 | "3x speedup/hour - 345x more efficient than alternatives" |

**Outcome:** FAILED - Made things 30% SLOWER

**Performance Numbers:**
| Benchmark | Before | After | Change |
|-----------|--------|-------|--------|
| g=8, K=7168 | 373 us | 488 us | -31% |
| g=8, K=2048 | 372 us | 462 us | -24% |
| g=2, K=4096 | 173 us | 249 us | -44% |
| g=2, K=1536 | 156 us | 228 us | -46% |

**Key Lessons:**
1. NVFP4 kernels are COMPUTE-BOUND, not memory-bound (4-bit data = tiny memory footprint)
2. TMA loads already complete quickly - no latency to hide
3. Adding pipeline stages added overhead without benefit
4. "Industry standard" optimizations don't apply universally

---

### ROUND 2: Tile Size Tuning

**Optimization Proposed:** Reduce tile size from (128, 128, 256) to (64, 128, 256) for better SM utilization.

**Shark Votes:**
| Shark | Vote | Score | Reasoning |
|-------|------|-------|-----------|
| Performance Oracle | Tile Size Tuning | 8.7/10 | "Addresses the actual bottleneck" |
| Pragmatic Engineer | Tile Size Tuning | 8.25/10 | "Removes waste instead of adding overhead" |
| ROI Maximizer | Tile Size Tuning | 8.5/10 | "Highest risk-adjusted ROI" |

**Outcome:** FAILED - COMPILE ERROR

**Error Message:**
```
OpError: expects the M-mode to be 128, but got 64
MmaMXF4NVF4Op error
```

**Key Lessons:**
1. NVFP4 MMA instruction (MmaMXF4NVF4Op) REQUIRES M-mode = 128
2. Hardware constraint: Cannot use smaller tiles for FP4 on Blackwell
3. Always check hardware requirements before proposing optimizations
4. Fixed configuration: `mma_tiler_mnk = (128, 128, 256)` CANNOT CHANGE

---

### ROUND 3: Wild Card Debug

**Optimization Proposed:** Investigate whether the kernel is computing the correct thing.

**Shark Votes:**
| Shark | Vote | Reasoning |
|-------|------|-----------|
| Performance Oracle | Wild Card | "Are we even solving the right problem?" |
| Pragmatic Engineer | Wild Card | "verify you're building the right thing" |
| ROI Maximizer | Wild Card | "costs 1 hour and could explain our 20-100x gap" |

**Outcome:** SUCCESS - Found the kernel was computing the WRONG thing

**Discovery:**
```python
# Task requires:
C = silu(A @ B1) * (A @ B2)  # TWO GEMMs with SiLU fusion

# Kernel computed:
C = A @ B  # Just ONE GEMM, no silu, no second gemm
```

**Key Lessons:**
1. The 20-100x performance gap was because we were measuring the WRONG computation
2. Before optimizing, VERIFY you're solving the right problem
3. Reading the spec carefully is the most valuable debugging tool

---

### ROUND 4: Minimal Fix (Two-Pass)

**Optimization Proposed:** Call the existing GEMM kernel twice, fuse SiLU in Python.

**Shark Votes:**
| Shark | Vote | Score | Reasoning |
|-------|------|-------|-----------|
| Performance Oracle | Minimal Fix | 7.9/10 | "Zero GPU changes = zero new bugs" |
| Pragmatic Engineer | Minimal Fix | 9.15/10 | "Implementation speed x correctness confidence = winner" |
| ROI Maximizer | Minimal Fix | INFINITE | "Correct output / 1 hour = INFINITE ROI" |

**Outcome:** SUCCESS - Implemented correctly

**Implementation:**
```python
# Pass 1: GEMM1 = A @ B1
run_single_gemm(a, b1, sfa_perm, sfb1_perm, temp1, problem_sizes)

# Pass 2: GEMM2 = A @ B2
run_single_gemm(a, b2, sfa_perm, sfb2_perm, temp2, problem_sizes)

# Fuse: C = silu(GEMM1) * GEMM2
result = silu(temp1.float()) * temp2.float()
```

**Key Lessons:**
1. "Correct but slow beats fast but wrong. Every. Single. Time."
2. Zero-risk approaches have value when past approaches have failed
3. A working baseline enables future optimization

---

### ROUND 5: Stream Parallelism

**Optimization Proposed:** Launch 8 groups on parallel CUDA streams instead of sequential.

**Shark Votes:**
| Shark | Vote | Score | Reasoning |
|-------|------|-------|-----------|
| Skeptic | Stream Parallelism | 6/10 | "Uses proven techniques, doesn't touch kernel" |
| Pragmatist | Stream Parallelism | 8/10 | "Fastest to test, zero kernel risk" |
| Theorist | Stream Parallelism | 7/10 | "Mathematically guaranteed improvement" |

**Outcome:** BLOCKED - GPUMode competition rules forbid multiple streams

**Error Message:**
```
"Your code contains work on another stream"
```

**Expected Impact (if allowed):**
| Benchmark | Current | Expected | Speedup |
|-----------|---------|----------|---------|
| g=8, K=7168 | ~530 us | ~80-180 us | 3-6x |
| g=8, K=2048 | ~508 us | ~70-170 us | 3-7x |

**Key Lessons:**
1. Competition rules matter - always check constraints
2. Stream parallelism WOULD be highly effective otherwise
3. Need to find single-stream optimizations

---

### ROUND 6: Reduce Launch Overhead

**Optimization Proposed:** Eliminate Python tensor creation overhead via pre-allocation.

**Shark Votes:**
| Shark | Vote | Score | Reasoning |
|-------|------|-------|-----------|
| Skeptic | Reduce Overhead | 8/10 | "Finally attacking the actual bottleneck" |
| Pragmatist | Reduce Overhead | 8/10 | "4-6 hours, low risk, 6-19x upside" |
| Theorist | Reduce Overhead | 8/10 | "The math checks out: 60us/group with 2-5us compute = 50us Python overhead" |

**Outcome:** FAILED - Made things 33% SLOWER

**The Hypothesis (incorrect):**
```python
# Identified overhead sources:
tensor_of_abc_ptrs = torch.tensor(abc_ptrs, device="cuda")  # ~15us
tensor_of_sfasfb_ptrs = torch.tensor(sfasfb_ptrs, device="cuda")  # ~15us
tensor_of_problem_sizes = torch.tensor(problem_sizes, device="cuda")  # ~15us
# Total: ~50us overhead per call
```

**Key Lessons:**
1. Python overhead wasn't the actual bottleneck
2. Pre-allocation added complexity without benefit
3. Need to look deeper at kernel execution

---

### ROUND 7: Research Edition

**Objective:** Deep investigation of kernel structure to find the real bottleneck.

**CRITICAL DISCOVERY: 75% OF THREADS ARE IDLE**

```python
# submission.py lines 316-354
if warp_idx == 0:  # ONLY WARP 0 DOES WORK!
    for k_tile in range(k_tile_cnt):
        # TMA loads
        # S2T copies
        # MMA operations  ALL COMPUTE HERE
```

**Thread Utilization:**
| Threads | Role | Status |
|---------|------|--------|
| 32 (Warp 0) | TMA + MMA + Everything | WORKING |
| 96 (Warps 1-3) | Nothing | **IDLE** |

**Root Cause Analysis:**
| Factor | Multiplier | Explanation |
|--------|------------|-------------|
| 75% threads idle | 4x | Only warp 0 works |
| No load/compute overlap | 1.5-2x | Single-stage pipeline |
| SIMT vs TMA stores | 1.2-1.5x | Inefficient epilogue |
| Two separate GEMMs | 1.3x | A matrix loaded twice |
| **Combined** | **9-24x** | Matches observed 25x gap |

**Key Lessons:**
1. The kernel is a REFERENCE IMPLEMENTATION, not optimized
2. The `if warp_idx == 0` pattern leaves 75% of hardware idle
3. cute.gemm() is a COOPERATIVE instruction - ALL threads should participate

---

### ROUND 8: Fix Warp Usage

**Optimization Proposed:** Enable all 128 threads to participate in MMA compute.

**Shark Votes:**
| Shark | Vote | Pitch | Reasoning |
|-------|------|-------|-----------|
| Skeptic | Pitch D | 9/10 | "Ask WHY before implementing WHAT" |
| Pragmatist | Pitch B | 8/10 | "Best risk/reward" |
| Theorist | Pitch A | 9/10 | "It's a BUG, not a feature" |

**Outcome:** IMPLEMENTED - READY TO TEST

**Code Changes:**
1. Consumer pipeline group changed from 1 thread to 128 threads
2. Main loop restructured:
   - TMA loads: Still warp 0 (hardware requirement)
   - S2T copies: Thread 0 only
   - **MMA compute: ALL threads** (the fix!)
   - Barrier added between S2T and MMA

**Expected Impact:**
| Metric | Before (v7) | After (v8) | Improvement |
|--------|-------------|------------|-------------|
| Active threads in MMA | 32 (25%) | 128 (100%) | 4x |
| Expected performance | 479 us | 120-240 us | 2-4x |

---

## 3. WHAT WORKED (with evidence)

### 1. Wild Card Debug (Round 3)
- **Technique:** Actually reading the task specification and comparing to kernel behavior
- **Impact:** Found fundamental bug - kernel computed A@B instead of silu(A@B1)*(A@B2)
- **Evidence:** Task.md clearly states `C = silu(A @ B1) * (A @ B2)`
- **Source:** `/shark_tank/rounds/round3_results.md`

### 2. Two-Pass Minimal Fix (Round 4)
- **Technique:** Call GEMM kernel twice, fuse SiLU in PyTorch
- **Impact:** Achieved correctness - validation passes with rtol=1e-03, atol=1e-03
- **Evidence:** Implementation in `/nvfp4_group_gemm/submission.py` lines 693-724
- **Source:** `/shark_tank/rounds/round4_results.md`

### 3. Research Investigation (Round 7)
- **Technique:** Deep code analysis to find hidden bottlenecks
- **Impact:** Discovered 75% of threads were idle (only warp 0 worked)
- **Evidence:** Code analysis showed `if warp_idx == 0` wrapping all compute
- **Source:** `/shark_tank/rounds/round7_results.md`

### 4. Warp Utilization Fix (Round 8)
- **Technique:** Move MMA compute outside warp 0 check, add proper barriers
- **Impact:** Pending verification - expected 2-4x speedup
- **Evidence:** Code changes in submission.py v8b
- **Source:** `/shark_tank/rounds/round8_implementation.md`

---

## 4. WHAT FAILED (with evidence)

### 1. Pipeline Stages (Round 1)
- **Technique:** Increase `num_ab_stage` from 1 to 3
- **Why it failed:** NVFP4 is compute-bound, not memory-bound. Adding stages added overhead.
- **Performance Impact:** 30-46% SLOWER
- **Evidence:** Round 1 results table shows all benchmarks regressed
- **Source:** `/shark_tank/rounds/round1_results.md`, `/SHARK_TANK_LEARNINGS.md`

### 2. Tile Size Tuning (Round 2)
- **Technique:** Reduce tile from (128,128,256) to (64,128,256)
- **Why it failed:** Hardware constraint - MmaMXF4NVF4Op requires M=128
- **Performance Impact:** Compile error, no execution
- **Evidence:** Error message "expects the M-mode to be 128, but got 64"
- **Source:** `/shark_tank/rounds/round2_results.md`

### 3. Triton Rewrite (Round 5 - not implemented)
- **Technique:** Rewrite kernel in Triton
- **Why it would fail:** Triton cannot access NVFP4 MMA hardware instructions
- **Performance Impact:** Would be ~10ms (20x WORSE) due to software FP4 decode
- **Source:** `/shark_tank/rounds/round5_results.md`

### 4. torch.compile (Round 5 - not implemented)
- **Technique:** Use torch.compile with max-autotune
- **Why it was risky:** Unverified if torch._scaled_mm supports FP4
- **Performance Impact:** Unknown, claims unverified
- **Source:** `/shark_tank/rounds/round5_results.md`

### 5. Stream Parallelism (Round 5)
- **Technique:** Run groups on parallel CUDA streams
- **Why it failed:** GPUMode competition rules forbid multiple streams
- **Performance Impact:** BLOCKED - would have been 4-7x if allowed
- **Evidence:** Error "Your code contains work on another stream"
- **Source:** `/shark_tank/rounds/round5_results.md`

### 6. Pre-allocation Overhead Reduction (Round 6)
- **Technique:** Cache metadata tensors to eliminate torch.tensor() overhead
- **Why it failed:** Python overhead wasn't the actual bottleneck
- **Performance Impact:** 33% SLOWER
- **Source:** `/shark_tank/rounds/round6_results.md`, `/shark_tank/rounds/round8_results.md`

---

## 5. PERFORMANCE TIMELINE

| Round | Approach | Before (us) | After (us) | Delta | Status |
|-------|----------|-------------|------------|-------|--------|
| 0 | Baseline (single GEMM) | N/A | ~373 | - | WRONG KERNEL |
| 1 | Pipeline Stages (3x) | 373 | 488 | -31% | FAILED |
| 2 | Tile Size (64x128) | 488 | - | COMPILE ERROR | FAILED |
| 3 | Wild Card Debug | 488 | - | Found bug | SUCCESS |
| 4 | Two-Pass Fix | ~488 | ~479 | Correct | SUCCESS |
| 5 | Stream Parallelism | 479 | - | BLOCKED | BLOCKED |
| 6 | Pre-allocation | 479 | ~640 | -33% | FAILED |
| 7 | Research | 479 | - | Found 75% idle | SUCCESS |
| 8 | All-Warp MMA | 479 | 120-240 (expected) | 2-4x (expected) | PENDING |

**Speed of Light Targets (from task.md):**
| M | N | K | L | Target (us) |
|---|---|---|---|-------------|
| 256 | 4096 | 7168 | 1 | 4.708 |
| 512 | 4096 | 7168 | 1 | 8.714 |
| 256 | 3072 | 4096 | 1 | 2.125 |
| 512 | 3072 | 7168 | 1 | 6.535 |

**Gap Analysis:** Current ~479us vs Target ~5us = ~100x gap

---

## 6. ARCHITECTURAL DECISIONS

### Why FP4 (e2m1)? Why not FP8?

**Answer:** Competition requirement - task.md specifies NVFP4 (e2m1) format.

**Technical Details:**
- Data type: `torch.float4_e2m1fn_x2` (packed, 2 values per byte)
- Values: [-1.5, -1, -0.5, 0, +0.5, +1, +1.5]
- Block scaling: 16 elements per scale factor
- Scale factor type: `fp8 (e4m3fnuz)`

**Why it matters:** FP4 has 4-bit elements vs FP8's 8-bit, reducing memory bandwidth by 2x but requiring specialized MMA instructions.

### Why CuTe DSL?

**Answer:** CuTe (from CUTLASS) provides:
1. Direct access to NVFP4 MMA instructions (`MmaMXF4NVF4Op`)
2. TMA (Tensor Memory Accelerator) integration for efficient memory loads
3. TMEM (Tensor Memory) management for Blackwell architecture
4. Built-in pipeline abstractions for producer/consumer patterns

**Source:** `/nvfp4_group_gemm/submission.py`, uses `cutlass.cute` module extensively.

### Why 128x128 Tiles?

**Answer:** Hardware constraint - not a choice.

**Evidence:**
```
OpError: expects the M-mode to be 128, but got 64
MmaMXF4NVF4Op error
```

The NVFP4 MMA instruction on Blackwell (SM 100) requires:
- M-mode = 128 (mandatory)
- N-mode = 128 (likely mandatory)
- K-mode = 64 per instruction, 256 per tile

**Configuration:** `mma_tiler_mnk = (128, 128, 256)` CANNOT CHANGE

### Why Single Pipeline Stage?

**Answer:** More stages made things SLOWER (Round 1 failure).

**Evidence:** 3-stage pipeline resulted in 30-46% performance regression.

**Reason:** NVFP4 is compute-bound due to tiny memory footprint of 4-bit data. Memory latency hiding via pipelining adds overhead without benefit.

### Why Two-Pass Instead of Fused Dual-GEMM?

**Answer:** Risk management after multiple failures.

**Rationale:**
1. Two-pass has zero GPU kernel changes = zero new bugs
2. Provides correct baseline for future optimization
3. "Correct but slow beats fast but wrong"

**Future opportunity:** True dual-GEMM fusion could save 1.3x by loading A matrix once instead of twice.

---

## 7. DEAD CODE / ORPHANED FILES

### Confirmed Dead/Superseded

| Path | Why Dead | Recommendation |
|------|----------|----------------|
| `/python/submission_triton.py` | Triton cannot access NVFP4 MMA | Archive - educational value |
| `/python/submission_streams.py` | Competition forbids streams | Archive - would work elsewhere |
| `/nvfp4_group_gemm/submission_v6_clean.py` | Superseded by v7, v8 | Archive |
| `/python/submission_prealloc.py` | Pre-allocation didn't help | Archive |

### Experimental (Unknown Status)

| Path | Notes | Recommendation |
|------|-------|----------------|
| `/python/submission_best*.py` | Multiple PyTorch attempts | Review - may have useful patterns |
| `/python/submission_compile.py` | torch.compile attempt | Test - may still be useful |
| `/python/submission_fused.py` | Fusion attempt | Review for dual-GEMM ideas |
| `/python/submission_cuda.py` | Custom CUDA attempt | Review |
| `/python/submission_inline_cuda.py` | Inline CUDA attempt | Review |

### Potentially Useful

| Path | Notes | Recommendation |
|------|-------|----------------|
| `/src/silu_mul_kernel.cu` | Fused SiLU kernel | May be useful for epilogue fusion |
| `/src/epilogue/left_silu_and_mul.h` | Custom epilogue | May be useful for true dual-GEMM |
| `/nvfp4_group_gemm/submission_v8_prealloc.py` | Pre-alloc version | Keep as reference for what didn't work |

---

## 8. OPEN QUESTIONS / GAPS

### Unanswered Questions

1. **Round 8 Results:** Has the all-warp MMA fix been tested? What's the actual performance?

2. **Dual-GEMM Fusion:** When will the kernel be upgraded to true dual-GEMM (load A once, compute both GEMMs)?

3. **TMA Store Epilogue:** Has TMA store in epilogue been tried? Expected 20-30% improvement.

4. **Scale Factor Conversion:** Is the `to_blocked()` conversion happening at runtime? Could it be eliminated?

5. **Competition Leaderboard:** What's the current position? What's the leader achieving?

### Missing Information

1. **Actual Benchmark Results:** Many rounds show "TBD" or "expected" numbers - need actual measurements.

2. **Nsight Profiling Data:** `/docs/gap6_nsight_proof.md` mentioned but detailed profiles not in results.

3. **cuBLAS Comparison:** How does the kernel compare to cuBLAS FP4 GEMM?

4. **Memory Bandwidth Analysis:** What's the actual memory throughput achieved?

5. **Register Pressure Analysis:** Are we hitting register limits with any configuration?

### Architectural Gaps

1. **True Dual-GEMM Fusion:**
   - Load A tiles once
   - Compute A@B1 and A@B2 in same mainloop
   - Fuse SiLU + multiply in epilogue
   - Expected: 1.3x improvement from halving A bandwidth

2. **Warp Specialization:**
   - Producer warps (TMA loads)
   - Consumer warps (MMA compute)
   - Expected: Additional overlap benefit

3. **TMA Store Epilogue:**
   - Replace SIMT 16-bit stores with TMA bulk stores
   - Expected: 20-30% improvement

4. **128-bit Vector Stores:**
   - Vectorize epilogue stores
   - Expected: 10-20% improvement

---

## Appendix: Key Code Snippets

### Task Specification (from task.md)
```python
C = silu(A @ B1) * (A @ B2)

# Input tensors:
# a: M x K x L in K-major order (nvfp4 e2m1)
# b1, b2: N x K x L in K-major order (nvfp4 e2m1)
# sfa: M x (K/16) x L scale factors (fp8 e4m3fnuz)
# sfb1, sfb2: N x (K/16) x L scale factors (fp8 e4m3fnuz)
# c: M x N x L output (fp16)
```

### Current Kernel Configuration (from submission.py)
```python
mma_tiler_mnk = (128, 128, 256)  # CANNOT CHANGE - hardware requirement
num_ab_stage = 2  # Double buffering for Round 8
threads_per_cta = 128
ab_dtype = cutlass.Float4E2M1FN
sf_dtype = cutlass.Float8E4M3FN
c_dtype = cutlass.Float16
sf_vec_size = 16
```

### Round 8 Fix Pattern (from round8_implementation.md)
```python
# AFTER: All threads participate in MMA
for k_tile in range(k_tile_cnt):
    # Warp 0 ONLY: TMA loads (hardware requirement)
    if warp_idx == 0:
        cute.copy(tma_atom_a, ...)
        cute.copy(tma_atom_b, ...)

    # ALL THREADS: Wait for TMA data
    ab_full = ab_consumer.wait_and_advance()

    # Thread 0 ONLY: S2T copy scale factors
    if tidx == 0:
        cute.copy(tiled_copy_s2t_sfa, ...)
        cute.copy(tiled_copy_s2t_sfb, ...)

    # BARRIER: Sync before MMA
    cute.arch.barrier()

    # ALL 128 THREADS: MMA compute (the fix!)
    for kblock_idx in range(num_kblocks):
        cute.gemm(tiled_mma, tCtAcc, tCrA, tCrB, tCtAcc)
```

---

## Summary of Key Learnings

1. **"Industry standard" optimizations don't apply universally** - Pipeline stages hurt FP4 kernels.

2. **Hardware constraints are absolute** - NVFP4 MMA requires 128x128 tiles, period.

3. **Verify before optimizing** - The kernel was wrong for 3 rounds before anyone checked.

4. **Correct but slow beats fast but wrong** - The two-pass approach was the right call.

5. **Competition rules matter** - Streams would have helped 4-7x but were forbidden.

6. **Look for idle resources** - 75% of threads were doing nothing.

7. **Cooperative instructions need cooperation** - `cute.gemm()` needs ALL threads.

8. **The best optimization is using hardware you already have** - Not buying new features.

---

*Report generated by the Archaeologist Agent*
*Date: 2026-01-25*
