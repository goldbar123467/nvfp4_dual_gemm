# SHARK TANK ROUND 8: FIX THE 75% IDLE THREADS

---

```
 _____ _____  __  __   _____ _   _ _____
|  ___|_ _\ \/ / |  \/  |  ___| | | | ____|
| |_   | | \  /  | |\/| | |_  | | | |  _|
|  _|  | | /  \  | |  | |  _| | |_| | |___
|_|   |___/_/\_\ |_|  |_|_|    \___/|_____|

 _____ _   _ ____  _____    _    ____  ____
|_   _| | | |  _ \| ____|  / \  |  _ \/ ___|
  | | | |_| | |_) |  _|   / _ \ | | | \___ \
  | | |  _  |  _ <| |___ / ___ \| |_| |___) |
  |_| |_| |_|_| \_\_____/_/   \_\____/|____/
```

---

## THE PROBLEM (From Round 7 Research)

```python
# Current submission.py lines 316-354
if warp_idx == 0:  # ← DISASTER: ONLY 25% OF THREADS WORK
    for k_tile in range(k_tile_cnt):
        # TMA loads
        # S2T copies
        # MMA operations
        # EVERYTHING on Warp 0 only
```

**128 threads available, 32 used. 75% idle.**

---

## THE GOAL

Make ALL 4 warps contribute to the computation.

---

## CONSTRAINTS

- Must use CuTe DSL (not raw CUDA)
- Must maintain correctness (rtol=1e-3)
- Cannot use multiple CUDA streams
- MMA tiles must be 128x128 minimum
- Cannot break the existing pipeline infrastructure

---

## APPROACHES TO CONSIDER

### Approach A: Cooperative Warps (All warps do same work)
- Remove `if warp_idx == 0` condition
- All warps participate in MMA
- Need to coordinate shared memory access

### Approach B: Warp Specialization (Producer/Consumer)
- Warp 0 = Producer (TMA loads only)
- Warps 1-3 = Consumers (MMA compute only)
- Use barriers to synchronize

### Approach C: Tile Partitioning
- Each warp handles different output tiles
- Warp 0: tiles 0-3, Warp 1: tiles 4-7, etc.
- Independent work, merge at end

### Approach D: K-Dimension Partitioning
- Split K loop across warps
- Each warp computes partial sum
- Reduce at the end

---

## WHAT THE CODE CURRENTLY DOES

The key insight is that the CuTe `tiled_mma` operations and pipeline are designed for single-warp execution in this reference implementation. We need to understand what can be safely parallelized.

**Safe to parallelize:**
- MMA operations (if we partition the work)
- Epilogue stores (already uses all threads)

**NOT safe to parallelize without changes:**
- TMA loads (uses pipeline barriers)
- S2T copies (writes to shared TMEM)
- Pipeline acquire/release (warp-specific)

---

## SUCCESS CRITERIA

| Metric | Current | Target |
|--------|---------|--------|
| Thread utilization | 25% | 75-100% |
| Performance | 479 µs | <200 µs |
| Correctness | ✓ | ✓ |

---

*"The best thread is one that's actually doing work."*

