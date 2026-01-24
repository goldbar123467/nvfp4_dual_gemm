# CONTEXT FOR NEXT ROUND

## TL;DR - ROUND 4 FIX IMPLEMENTED

**Task requires:** `C = silu(A @ B1) * (A @ B2)` (dual GEMM with SiLU fusion)
**Round 4 fix:** Two-Pass approach - call single GEMM twice, fuse in PyTorch

---

## ROUND 4 IMPLEMENTATION

The `solve()` function in submission.py now implements:

```python
def solve(data: input_t) -> output_t:
    # Unpack: (a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c)
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c = data

    # Pass 1: GEMM1 = A @ B1
    run_single_gemm(a, b1, sfa_perm, sfb1_perm, temp1, problem_sizes)

    # Pass 2: GEMM2 = A @ B2
    run_single_gemm(a, b2, sfa_perm, sfb2_perm, temp2, problem_sizes)

    # Fuse: C = silu(GEMM1) * GEMM2
    result = silu(temp1.float()) * temp2.float()
    c.copy_(result.half())
```

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

## NEXT OPTIMIZATIONS (After Validation)

1. **Interleaved Dual GEMM**: Compute both in mainloop, reuse A tiles
2. **Fused Epilogue**: Move silu + multiply into GPU kernel
3. **TMA Store Epilogue**: Replace SIMT stores with TMA hardware
4. **Warp Specialization**: Producer/consumer architecture

---

## FILES MODIFIED

- `/home/clark/nvfp4_dual_gemm_repo/nvfp4_group_gemm/submission.py` - UPDATED
- `/home/clark/nvfp4_group_gemm/submission.py` - COPIED

---

## SHARK TANK SCORE

| Round | Winner | Expected | Actual |
|-------|--------|----------|--------|
| 1 | Pipeline Stages | 1.5x faster | 30% SLOWER |
| 2 | Tile Tuning | 2-3x faster | COMPILE ERROR |
| 3 | Wild Card | ??? | Found the bug |
| 4 | Minimal Fix | Correctness | IMPLEMENTED |

**Status:** Awaiting user testing
