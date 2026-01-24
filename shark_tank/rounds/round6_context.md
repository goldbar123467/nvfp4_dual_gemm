# SHARK TANK ROUND 6: THE PERFORMANCE HUNT

---

```
 ____   ___  _   _ _   _ ____     __
|  _ \ / _ \| | | | \ | |  _ \   / /_
| |_) | | | | | | |  \| | | | | | '_ \
|  _ <| |_| | |_| | |\  | |_| | | (_) |
|_| \_\\___/ \___/|_| \_|____/   \___/

  _____ _   _ _____   _   _ _   _ _   _ _____
 |_   _| | | | ____| | | | | | | | \ | |_   _|
   | | | |_| |  _|   | |_| | | | |  \| | | |
   | | |  _  | |___  |  _  | |_| | |\  | | |
   |_| |_| |_|_____| |_| |_|\___/|_| \_| |_|
```

---

## THE SITUATION

After 5 rounds:
- Streams: NOT ALLOWED (competition rules)
- Pipeline stages: SLOWER (compute-bound)
- Tile tuning: IMPOSSIBLE (hardware constraint)
- We're still **25-145x off target**

**Current Performance:**
```
g=8, K=7168: 479 µs (target: 18.8 µs) - 25x gap
g=8, K=2048: 451 µs (target: 10.7 µs) - 42x gap
g=2, K=4096: 247 µs (target: 2.4 µs)  - 103x gap
g=2, K=1536: 218 µs (target: 1.5 µs)  - 145x gap
```

---

## THE BIG QUESTION

Why are we so far off? Let's analyze:

**Per-group timing (g=8 case):**
- Total: 479 µs / 8 groups = ~60 µs per group
- But individual GEMMs should be ~2-5 µs each

**Hypothesis:** The overhead is NOT in the GEMM computation itself, but in:
1. Kernel launch overhead (repeated compilations?)
2. TMA descriptor setup per group
3. CTA scheduling inefficiency
4. Memory allocation/deallocation

---

## HARDWARE CONSTRAINTS (STILL APPLY)

```python
# CANNOT CHANGE - NVFP4 MMA requires 128x128
mma_tiler_mnk = (128, 128, 256)

# CANNOT USE - Multiple CUDA streams
torch.cuda.Stream()  # FORBIDDEN

# MAKES THINGS WORSE
num_ab_stage = 3  # Pipeline stages = SLOWER
```

---

## WHAT WE'RE OPTIMIZING

GROUP GEMM: Multiple independent GEMMs batched together
- Each group: C = A @ B (with block scaling)
- Different M sizes per group (64-384)
- Same K and N within benchmark

Current approach: Single kernel launch, all groups in Z-dimension

---

## ROUND 6 OPTIMIZATION APPROACHES

### Contestant 1: PERSISTENT KERNEL
Keep thread blocks alive, process multiple groups without relaunching.

### Contestant 2: KERNEL FUSION + WARP SPECIALIZATION
Producer warps load data, consumer warps compute MMA.

### Contestant 3: SPLIT-K PARALLELISM
Partition K dimension across CTAs, reduce at the end.

### Contestant 4: TMA PREFETCH OPTIMIZATION
Overlap TMA descriptor updates with computation.

---

## JUDGING CRITERIA

1. **Feasibility** - Can it actually be implemented in CuTe DSL?
2. **Expected Speedup** - How much faster, with math to back it up
3. **Risk Level** - What could go wrong?
4. **Competition Rules** - Does it violate any constraints?

---

## WHAT FAILED (DON'T TRY AGAIN)

| Optimization | Result | Reason |
|--------------|--------|--------|
| `num_ab_stage = 3` | 30% SLOWER | Compute-bound, not memory-bound |
| `mma_tiler_mnk = (64,128,256)` | COMPILE ERROR | Hardware requires M=128 |
| Multiple CUDA streams | FORBIDDEN | Competition rules |

---

## THE STAKES

We need a **25x speedup minimum** just to hit target on the easiest benchmark.
Conservative approaches have consistently FAILED.

---

*"When you're 100x off target, you don't need a 10% improvement—you need a paradigm shift."*

