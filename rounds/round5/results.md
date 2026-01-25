# SHARK TANK ROUND 5: ALL WILD CARD EDITION - RESULTS

---

```
 __        _____ _     ____     ____    _    ____  ____
 \ \      / /_ _| |   |  _ \   / ___|  / \  |  _ \|  _ \
  \ \ /\ / / | || |   | | | | | |     / _ \ | |_) | | | |
   \ V  V /  | || |___| |_| | | |___ / ___ \|  _ <| |_| |
    \_/\_/  |___|_____|____/   \____/_/   \_\_| \_\____/

        __        _____ _   _ _   _ _____ ____  _
        \ \      / /_ _| \ | | \ | | ____|  _ \| |
         \ \ /\ / / | ||  \| |  \| |  _| | |_) | |
          \ V  V /  | || |\  | |\  | |___|  _ <|_|
           \_/\_/  |___|_| \_|_| \_|_____|_| \_(_)
```

---

## THE PITCHES

### WILD CARD A: TRITON KERNEL
*"Let's rewrite in Triton!"*

**The Honest Confession:**
> "Triton is NOT the right tool for this job."
> - Wild Card A, in their own pitch

**Why it failed:**
- Triton cannot access NVFP4 MMA hardware (`MmaMXF4NVF4Op`)
- Would need software FP4 decode (~10ms - 20x WORSE)
- Even the epilogue-only variant saves just 10-30 us

**Scores:** Skeptic: 3/10 | Pragmatist: 3/10 | Theorist: 2/10

---

### WILD CARD B: TORCH.COMPILE
*"Let the compiler do the work!"*

**The Promise:**
> "Claims 10-50x speedup"

**The Problem:**
> "IF torch._scaled_mm supports FP4..." (unverified)

**Why it's risky:**
- Hinges on unverified PyTorch internal support
- Scale factor format conversion is handwaved
- After Round 3, we don't trust unverified assumptions

**Scores:** Skeptic: 4/10 | Pragmatist: 5/10 | Theorist: 5/10

---

### WILD CARD C: STREAM PARALLELISM
*"While everyone else is tuning the engine, I'm adding more engines."*

**The Key Insight:**
```
Current:  8 groups processed SEQUENTIALLY
Proposed: 8 groups processed in PARALLEL

B200 has 192 SMs
8 groups = ~512 CTAs total
512 / 192 = 2.7 waves (vs 8 sequential waves)

Expected speedup: 4-7x
```

**Why it won:**
- Mathematics are VERIFIABLE
- Doesn't touch the kernel (zero bug risk)
- Uses proven technique (streams + CUDA graphs)
- Attacks a REAL inefficiency (sequential on parallel hardware)

**Scores:** Skeptic: 6/10 | Pragmatist: 8/10 | Theorist: 7/10

---

## THE VOTE

```
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  ROUND 5 WINNER: WILD CARD C - STREAM PARALLELISM              ║
║                                                                ║
║  UNANIMOUS VOTE (3-0)                                          ║
║                                                                ║
║  Shark 1 (Skeptic):   "Uses proven techniques, doesn't         ║
║                        touch the kernel. Realistic claims."    ║
║                                                                ║
║  Shark 2 (Pragmatist): "Fastest to test, zero kernel risk,    ║
║                         conservative claims are believable."   ║
║                                                                ║
║  Shark 3 (Theorist):   "Mathematically guaranteed improvement. ║
║                         Attacks fundamental inefficiency."     ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

## WHY THIS COULD ACTUALLY WORK

After 4 rounds of failure, what makes this different?

| Previous Attempts | Why They Failed | Stream Parallelism |
|-------------------|-----------------|-------------------|
| Pipeline stages | Compute-bound, not memory-bound | ORTHOGONAL - different dimension |
| Tile sizes | Hardware constraint (128x128) | DOESN'T CHANGE TILES |
| Wrong kernel | Was computing A@B instead of silu(A@B1)*A@B2 | DOESN'T TOUCH KERNEL |

**This is attacking a DIFFERENT dimension of the problem.**

---

## EXPECTED IMPACT

| Benchmark | Current | Expected | Speedup |
|-----------|---------|----------|---------|
| g=8, K=7168 | ~530 us | ~80-180 us | 3-6x |
| g=8, K=2048 | ~508 us | ~70-170 us | 3-7x |
| g=2, K=4096 | ~279 us | ~150-200 us | 1.5-2x |
| g=2, K=1536 | ~256 us | ~130-180 us | 1.5-2x |

Note: Smaller groups (g=2) benefit less because less parallelism to exploit.

---

## IMPLEMENTATION PLAN

1. Create stream pool (one per group or shared pool)
2. Launch each group's GEMM on separate stream
3. Optionally wrap in CUDA graph for repeated execution
4. Synchronize at the end

**Time to implement:** 1-2 hours
**Risk:** Low (doesn't modify kernel)
**Worst case:** No change (streams have overhead)

---

## SEASON SCORECARD

| Round | Winner | Expected | Actual | Status |
|-------|--------|----------|--------|--------|
| 1 | Pipeline Stages | 1.5x faster | 30% SLOWER | FAILED |
| 2 | Tile Size Tuning | 2-3x faster | COMPILE ERROR | FAILED |
| 3 | Wild Card | ??? | Found the bug | SUCCESS |
| 4 | Minimal Fix | Correctness | Fixed kernel | SUCCESS |
| 5 | **Stream Parallelism** | **4-7x faster** | **TBD** | **IN PROGRESS** |

---

*"The best optimization isn't making one thing faster—it's doing many things at once."*
*- Wild Card C*
