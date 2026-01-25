# ARCHIVIST MANIFEST: Shark Tank Knowledge Storage

**Status:** BLOCKED - RAG Brain Unavailable
**Date:** 2026-01-25
**Error:** Ollama connection error - embedding service not responding

---

## Storage Attempt Summary

The Archivist attempted to store 22+ memories but the RAG brain's Ollama embedding service is down. All memories are documented below for manual storage or retry when service is restored.

---

## ROUND OUTCOMES (8 memories, category: "outcome")

### Round 1: Pipeline Stages
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** outcome
- **Tags:** round-1, shark-tank, optimization, pipeline-stages, failure
- **Summary:** Pipeline stages made things 30% SLOWER. NVFP4 is compute-bound, not memory-bound. Adding stages added overhead.
- **Performance:** 373us -> 488us (-31%)

### Round 2: Tile Size Tuning
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** outcome
- **Tags:** round-2, shark-tank, optimization, tile-size, failure, hardware-constraint
- **Summary:** COMPILE ERROR - MmaMXF4NVF4Op requires M-mode = 128. Hardware constraint prevents smaller tiles.
- **Error:** "expects the M-mode to be 128, but got 64"

### Round 3: Wild Card Debug
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** outcome
- **Tags:** round-3, shark-tank, optimization, wild-card, success, bug-discovery
- **Summary:** SUCCESS - Found kernel was computing A@B instead of silu(A@B1)*(A@B2). The entire kernel was WRONG.

### Round 4: Minimal Fix (Two-Pass)
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** outcome
- **Tags:** round-4, shark-tank, optimization, two-pass, success, correctness
- **Summary:** SUCCESS - Two-pass implementation achieved correctness. "Correct but slow beats fast but wrong."

### Round 5: Stream Parallelism
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** outcome
- **Tags:** round-5, shark-tank, optimization, streams, blocked, competition-rules
- **Summary:** BLOCKED - GPUMode competition rules forbid multiple CUDA streams. Would have achieved 3-7x speedup.

### Round 6: Reduce Launch Overhead
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** outcome
- **Tags:** round-6, shark-tank, optimization, pre-allocation, failure, wrong-bottleneck
- **Summary:** FAILED - Pre-allocation made things 33% SLOWER. Python overhead wasn't the actual bottleneck.

### Round 7: Research Edition
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** outcome
- **Tags:** round-7, shark-tank, optimization, research, success, thread-utilization, warp-idle
- **Summary:** SUCCESS - Discovered 75% of threads (warps 1-3) were IDLE. Only warp 0 was working.

### Round 8: Fix Warp Usage
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** outcome
- **Tags:** round-8, shark-tank, optimization, warp-utilization, pending, all-threads-mma
- **Summary:** IMPLEMENTED - All 128 threads now participate in MMA. Expected 2-4x speedup. Awaiting test results.

---

## TECHNICAL DISCOVERIES (6 memories, category: "insight")

### 1. NVFP4 is compute-bound, not memory-bound
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** insight
- **Tags:** technical, gpu, blackwell, fp4, compute-bound
- **Content:** FP4 data (4-bit) has tiny memory footprint. TMA loads complete quickly. Memory latency hiding provides no benefit.

### 2. MMA instruction requires 128x128 tiles
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** insight
- **Tags:** technical, gpu, blackwell, fp4, hardware-constraint, mma
- **Content:** MmaMXF4NVF4Op on Blackwell (SM 100) requires M-mode = 128, N-mode = 128. Configuration mma_tiler_mnk = (128, 128, 256) CANNOT CHANGE.

### 3. The kernel was fundamentally wrong
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** insight
- **Tags:** technical, gpu, bug, correctness
- **Content:** Kernel computed single GEMM (A@B) instead of dual GEMM with fusion (silu(A@B1)*(A@B2)). 3 rounds wasted on wrong kernel.

### 4. 75% of threads were idle
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** insight
- **Tags:** technical, gpu, thread-utilization, warp
- **Content:** Code pattern "if warp_idx == 0:" left warps 1-3 (96 threads) idle. Only 32 threads doing work. cute.gemm() is COOPERATIVE.

### 5. Competition forbids multiple CUDA streams
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** insight
- **Tags:** technical, gpu, competition, constraints
- **Content:** GPUMode competition rules explicitly forbid using multiple CUDA streams. Error: "Your code contains work on another stream"

### 6. Pre-allocation made things slower
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** insight
- **Tags:** technical, gpu, overhead, wrong-assumption
- **Content:** Hypothesis that torch.tensor() calls caused 50us overhead was INCORRECT. Pre-allocation added 33% slowdown.

---

## ANTI-PATTERNS (4 memories, category: "pattern")

### 1. Pipeline stages on compute-bound kernels
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** pattern (anti-pattern)
- **Tags:** anti-pattern, failure, gpu, pipeline
- **Content:** DO NOT add pipeline stages to compute-bound kernels. FP4 has tiny memory footprint - no latency to hide.

### 2. Tile sizes smaller than hardware minimum
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** pattern (anti-pattern)
- **Tags:** anti-pattern, failure, gpu, tile-size
- **Content:** DO NOT try tile sizes below hardware requirements. NVFP4 MMA requires 128x128. Check specs BEFORE optimizing.

### 3. Stream parallelism in constrained competitions
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** pattern (anti-pattern)
- **Tags:** anti-pattern, failure, gpu, streams, competition
- **Content:** DO NOT assume standard optimizations are allowed. Competition rules may forbid streams, specific APIs, etc.

### 4. Pre-allocation for overhead reduction (wrong bottleneck)
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** pattern (anti-pattern)
- **Tags:** anti-pattern, failure, gpu, overhead
- **Content:** DO NOT optimize Python overhead without profiling proof. The bottleneck was in GPU kernel, not tensor creation.

---

## SUCCESS PATTERNS (4 memories, category: "pattern")

### 1. Read the spec to find bugs
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** pattern (success)
- **Tags:** pattern, success, debugging, verification
- **Content:** Always verify kernel computes the CORRECT thing before optimizing. Round 3 found 3-round bug by reading task.md.

### 2. Two-pass approach for correctness first
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** pattern (success)
- **Tags:** pattern, success, correctness, risk-management
- **Content:** When past approaches failed, zero-risk two-pass (call kernel twice, fuse in Python) achieved correct baseline.

### 3. Research rounds to find hidden bottlenecks
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** pattern (success)
- **Tags:** pattern, success, research, investigation
- **Content:** Dedicated investigation round found 75% thread idle issue. Don't just try optimizations - understand the kernel.

### 4. Enable all warps for cooperative instructions
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** pattern (success)
- **Tags:** pattern, success, gpu, cooperative, mma
- **Content:** cute.gemm() is COOPERATIVE - ALL threads must participate. Moving MMA outside "if warp_idx == 0" enabled 4x threads.

---

## KEY DECISIONS (4 memories, category: "decision")

### 1. Why CuTe DSL
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** decision
- **Tags:** decision, architecture, cute, cutlass
- **Content:** CuTe provides direct access to NVFP4 MMA (MmaMXF4NVF4Op), TMA integration, TMEM management for Blackwell, and pipeline abstractions.

### 2. Why 128x128 tiles
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** decision
- **Tags:** decision, architecture, tile-size, hardware
- **Content:** Not a choice - hardware requirement. MmaMXF4NVF4Op requires M=128, N=128. Smaller tiles cause compile error.

### 3. Why two-pass instead of fused
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** decision
- **Tags:** decision, architecture, two-pass, risk-management
- **Content:** Risk management after multiple failures. Zero GPU kernel changes = zero new bugs. Provides correct baseline for future optimization.

### 4. Why single pipeline stage
- **Status:** FAILED TO STORE (Ollama down)
- **Category:** decision
- **Tags:** decision, architecture, pipeline, compute-bound
- **Content:** More stages made things SLOWER (Round 1). NVFP4 is compute-bound - memory latency hiding via pipelining adds overhead without benefit.

---

## PERFORMANCE BASELINE (1 memory, category: "documentation")

- **Status:** FAILED TO STORE (Ollama down)
- **Category:** documentation
- **Tags:** performance, baseline, gap-analysis
- **Content:**
  - Current performance: ~479us
  - Speed-of-light target: ~5us
  - Gap: ~100x
  - Round 8 expected: 120-240us (2-4x improvement)

---

## VERIFICATION TESTS

### Test Query 1: "What failed in Shark Tank?"
- **Status:** COULD NOT EXECUTE (Ollama down)
- **Expected Results:** Round 1 (pipeline stages), Round 2 (tile size), Round 6 (pre-allocation)

### Test Query 2: "NVFP4 hardware constraints"
- **Status:** COULD NOT EXECUTE (Ollama down)
- **Expected Results:** 128x128 tile requirement, MmaMXF4NVF4Op specs

### Test Query 3: "Round 3 discovery"
- **Status:** COULD NOT EXECUTE (Ollama down)
- **Expected Results:** Kernel was wrong - single GEMM instead of dual GEMM

---

## NEXT STEPS

1. **Restart Ollama service:** `systemctl start ollama` or check Docker container
2. **Retry storage:** Re-run Archivist agent once Ollama is up
3. **Manual import:** Use this manifest to manually store memories if needed

---

## RAW MEMORY CONTENT FOR RETRY

All memory content is documented above with:
- Project: shark-tank-gpumode
- Source: claude
- Categories and tags specified per memory

When Ollama is restored, the Archivist should re-attempt all 22+ memory stores.

---

*Manifest generated by the Archivist Agent*
*Date: 2026-01-25*
*Status: BLOCKED - Awaiting RAG Brain restoration*
