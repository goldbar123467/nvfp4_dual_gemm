# CONTEXT FOR NEXT ROUND

## TL;DR - THE KERNEL IS WRONG

**Task requires:** `C = silu(A @ B1) * (A @ B2)` (dual GEMM with SiLU fusion)
**Kernel computes:** `C = A @ B` (single GEMM, no silu, no second gemm)

This is why we're 20-100x off target.

---

## WHAT FAILED (DON'T TRY AGAIN)

| Optimization | Result | Reason |
|--------------|--------|--------|
| `num_ab_stage = 3` | 30% SLOWER | Compute-bound, not memory-bound |
| `mma_tiler_mnk = (64,128,256)` | COMPILE ERROR | Hardware requires M=128 |
| `mma_tiler_mnk = (128,64,256)` | COMPILE ERROR | Hardware requires N=128 |

---

## HARDWARE CONSTRAINTS

```python
# CANNOT CHANGE - NVFP4 MMA requires 128x128
mma_tiler_mnk = (128, 128, 256)
```

---

## WHAT NEEDS TO BE IMPLEMENTED

Current kernel mainloop (lines 314-352 in submission.py):
- Loads A, B, SFA, SFB
- Computes ONE gemm: `acc = A @ B`
- Stores result

Needed:
1. Load A, B1, B2, SFA, SFB1, SFB2
2. Compute acc1 = A @ B1
3. Compute acc2 = A @ B2 (reuse A from shared memory!)
4. Epilogue: C = silu(acc1) * acc2

---

## INPUT STRUCTURE (from task.md)

```python
# Input tuple:
(a, b1, b2, sfa, sfb1, sfb2, c)

# But current kernel only uses:
(a, b, sfa, sfb, c)  # Missing b2 and sfb2!
```

---

## BASELINE PERFORMANCE

```
g=8, K=7168: ~400-460 µs (target: 18.8 µs)
g=8, K=2048: ~400-440 µs (target: 10.7 µs)
g=2, K=4096: ~170-245 µs (target: 2.4 µs)
g=2, K=1536: ~150-222 µs (target: 1.5 µs)
```

---

## KEY INSIGHT

The "optimization" that will give us the biggest speedup is **actually implementing the correct algorithm**. Once we compute dual GEMM with SiLU, we can then optimize:
- TMA store epilogue (5-10%)
- Warp specialization (10-20%)
- Interleaved dual GEMM computation

---

## FILES TO MODIFY

- `/home/clark/nvfp4_dual_gemm_repo/nvfp4_group_gemm/submission.py`
- `/home/clark/nvfp4_group_gemm/submission.py` (test copy)

---

## SHARK TANK SCORE

- Round 1: Pipeline Stages - FAILED
- Round 2: Tile Tuning - FAILED
- Round 3: Wild Card - DISCOVERED THE BUG

**Round 4 Goal:** Implement the actual dual GEMM with SiLU fusion
