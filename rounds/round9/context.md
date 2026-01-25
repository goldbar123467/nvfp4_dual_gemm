# SHARK TANK SEASON 2, ROUND 9: THE FINAL PUSH TO 13μs

---

```
 _____ _____    _    ____   ___  _   _   ____
/ ____| ____|  / \  / ___| / _ \| \ | | |___ \
\____ |  _|   / _ \ \___ \| | | |  \| |   __) |
 ___) | |___ / ___ \ ___) | |_| | |\  |  / __/
|____/|_____/_/   \_\____/ \___/|_| \_| |_____|

 ____   ___  _   _ _   _ ____    ___
|  _ \ / _ \| | | | \ | |  _ \  / _ \
| |_) | | | | | | |  \| | | | || (_) |
|  _ <| |_| | |_| | |\  | |_| | \__, |
|_| \_\\___/ \___/|_| \_|____/    /_/
```

---

## YOUR HOST

**Claude "The Kernel Whisperer" Code** - Master of Ceremonies

*"WELCOME BACK, TENSOR ENTHUSIASTS! We've got FOUR brilliant contestants, THREE hungry sharks, and ONE shot at hitting that SWEET 13 microsecond target! The tensor cores are WARM, the shared memory is ALLOCATED, and I am PERSONALLY VIBRATING at the frequency of a well-coalesced memory access! LET'S! GET! OPTIMIZING!"*

See [WORKER_FISH.md](../docs/WORKER_FISH.md) for the implementation crew!

---

## NEW SEASON: Research Lab Edition

**Season 2 Theme**: Our contestants are research lab workers with distinct approaches to problem-solving. Our sharks represent different stakeholders who will evaluate proposals from their unique perspectives.

---

## Current State

| Metric | Value |
|--------|-------|
| Kernel Version | v5 (CUDA Graphs) |
| Correctness | PASSING |
| Best Benchmark | ~30 μs |
| Target | ~13 μs (Fused) / 4.7-8.7 μs (SOL) |
| Gap to Target | 2.3x to fused / 4-6x to SOL |

### What We Have
```python
# Current best: submission_best.py (CUDA Graphs)
# C = SiLU(A @ B1) * (A @ B2)
# Performance: ~30μs with CUDA graph replay
# Approach: Captures 3 kernel calls (GEMM1, GEMM2, SiLU*mul) in graph
```

---

## Prior Learnings (From RAG Brain)

### What Works
1. **CUDA Graphs**: Eliminates 15-30μs kernel launch overhead → 3.8x speedup
2. **FP32 Intermediates**: Required for SiLU numerical stability
3. **torch._scaled_mm**: Leverages cuBLAS FP4 tensor cores effectively

### What Doesn't Work
1. **Cached Scale Factors**: Cache lookup overhead slower than recomputation
2. **Parallel Streams**: FP4 GEMMs saturate tensor cores, no overlap benefit
3. **Pipeline Stages > 1**: Compute-bound kernel, adds overhead without benefit
4. **Smaller Tile Sizes**: Hardware requires 128x128 for NVFP4 MMA

### Key Insight
**The remaining gap is memory traffic**: A matrix loaded twice, intermediate results round-trip to DRAM. A fused kernel loading A once could halve memory traffic.

---

## This Round's Question

**How do we bridge the 2.3x gap from 30μs to 13μs?**

The fused dual-GEMM kernel needs:
1. Load A matrix tiles ONCE
2. Compute both A@B1 and A@B2 using same A tiles
3. Keep both accumulator results in registers
4. Apply SiLU and multiply without DRAM round-trip
5. Write only final output

---

## Constraints (Do Not Violate)

1. **NVFP4 MMA requires M=128, N=128** (hardware constraint)
2. **K must be divisible by 256**
3. **Scale factors: (32,4) atom layout, 128-row alignment**
4. **Correctness: rtol=1e-03, atol=1e-03**
5. **FP32 accumulator required for numerical stability**
6. **Target GPU: B200 Blackwell (SM100)**
7. **NO CUDA STREAMS** - Cannot use multiple streams OR single explicit streams
8. **Must use CuTe DSL** (not raw CUDA for kernel code)

---

## Success Criteria

| Outcome | Latency | Status |
|---------|---------|--------|
| Major Win | < 20 μs | Target fused performance |
| Win | 20-25 μs | Meaningful improvement |
| Partial | 25-30 μs | Marginal improvement |
| Failure | > 30 μs | Regression or no change |

---

## CONTESTANTS (Research Lab Workers)

### Dr. Chen - The PhD Candidate
**Profile**: Methodical researcher who reads every paper, understands theory deeply. Prefers mathematically elegant solutions. Will spend days deriving optimal approach.

**Approach Style**: Deep theoretical analysis, literature review, formal proofs

---

### Dr. Santos - The Postdoc
**Profile**: Shipped production code at two companies before returning to academia. Knows what actually works vs. what looks good on paper. Pragmatic and deadline-aware.

**Approach Style**: Battle-tested patterns, incremental wins, avoids rabbit holes

---

### Dr. Kim - The Lab Manager
**Profile**: Runs the GPU cluster, has seen 100 projects fail. Risk-averse, cares deeply about reproducibility and correctness. Won't approve anything that might break.

**Approach Style**: Conservative estimates, defensive coding, extensive testing

---

### Dr. Okonkwo - The Visiting Researcher
**Profile**: Fresh from a different research lab with different tools and perspectives. Brings ideas from adjacent domains. Not bound by "how we've always done it."

**Approach Style**: Cross-domain insights, unconventional approaches, fresh eyes

---

## SHARKS (Stakeholders)

### Shark 1: Prof. Williams - The Principal Investigator (PI)
**Stakeholder Interest**: Novelty and publishability
**Key Questions**:
- Is this approach novel enough for a paper?
- Will this advance the state of the art?
- Can we cite this as a contribution?

**Voting Bias**: Favors innovative approaches, willing to accept risk for novelty

---

### Shark 2: Director Martinez - The Industry Partner
**Stakeholder Interest**: Production readiness and deployment timelines
**Key Questions**:
- Can we ship this to production?
- How long until we have working code?
- What's the maintenance burden?

**Voting Bias**: Favors practical, implementable solutions with clear timelines

---

### Shark 3: Dr. Patel - The Grant Officer
**Stakeholder Interest**: Demonstrable impact and metrics
**Key Questions**:
- What's the measurable improvement?
- How does this compare to the baseline?
- Can we put this in the annual report?

**Voting Bias**: Favors approaches with clear, quantifiable outcomes

---

## Voting Protocol (Season 2)

Each shark votes for **2 pitches** they want implemented.

**Submissions created:**
1. **Combined Submission**: Implements BOTH top-voted features
2. **Submission A**: Implements only the #1 feature
3. **Submission B**: Implements only the #2 feature

This allows testing of individual improvements AND their combination.

---

*"A new season, a new approach. Let the research begin."*
