# PITCH A: Triton Fused Dual-GEMM Kernel

**Contestant**: Dr. Chen (PhD Candidate)

---

## The Theoretical Foundation

After reviewing 47 papers on GEMM optimization and the Triton documentation, I've identified that our current CUDA Graphs approach is fundamentally limited by **kernel granularity boundaries**.

### Memory Traffic Analysis

```
Current (3 kernels):
┌─────────────────────────────────────────────────────┐
│ Kernel 1 (GEMM1):                                   │
│   Read:  A (M×K/2 bytes) + B1 (N×K/2 bytes)        │
│   Write: R1 (M×N×4 bytes)                          │
│                                                     │
│ Kernel 2 (GEMM2):                                   │
│   Read:  A (M×K/2 bytes) + B2 (N×K/2 bytes)        │  ← A loaded TWICE
│   Write: R2 (M×N×4 bytes)                          │
│                                                     │
│ Kernel 3 (Epilogue):                               │
│   Read:  R1 + R2 (M×N×8 bytes)                     │  ← R1,R2 round-trip
│   Write: C (M×N×2 bytes)                           │
└─────────────────────────────────────────────────────┘

For M=512, N=4096, K=7168:
  A reads: 2 × (512 × 7168 / 2) = 3.67 MB
  R1,R2:   2 × (512 × 4096 × 4) = 16.8 MB round-trip
  Total extra traffic: ~20 MB
```

### Theoretical Minimum

```
Fused (1 kernel):
┌─────────────────────────────────────────────────────┐
│ Fused Kernel:                                       │
│   Read:  A (ONCE) + B1 + B2 + scales               │
│   Write: C only                                     │
│   (R1, R2 stay in registers/shared memory)         │
└─────────────────────────────────────────────────────┘

Traffic reduction: ~20 MB → 0 MB intermediate
Expected speedup: 30μs × (1 - 20/30) ≈ 10μs saved → 20μs
```

---

## Proposed Solution: Triton Fused Kernel

I propose implementing a **Triton kernel** that fuses both GEMMs and the epilogue into a single GPU kernel.

### Why Triton?

1. **Block-level programming**: Natural fit for tiled GEMM
2. **Auto-tuning**: Finds optimal tile sizes automatically
3. **Python syntax**: Faster iteration than CUTLASS/CUDA
4. **FP4 support**: Triton 3.0+ supports block-scaled FP4 on Blackwell

### Kernel Design

```python
@triton.jit
def fused_dual_gemm_silu_kernel(
    A_ptr, B1_ptr, B2_ptr, C_ptr,
    sf_a_ptr, sf_b1_ptr, sf_b2_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program ID determines output tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Initialize TWO accumulators
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop: Load A ONCE, use for both GEMMs
    for k in range(0, K, BLOCK_K):
        # Load A tile (SHARED between both computations)
        a_tile = load_fp4_block_scaled(A_ptr, sf_a_ptr, ...)

        # Load B1 tile, compute GEMM1
        b1_tile = load_fp4_block_scaled(B1_ptr, sf_b1_ptr, ...)
        acc1 += tl.dot(a_tile, b1_tile.T)

        # Load B2 tile, compute GEMM2 (reuse a_tile!)
        b2_tile = load_fp4_block_scaled(B2_ptr, sf_b2_ptr, ...)
        acc2 += tl.dot(a_tile, b2_tile.T)

    # Fused epilogue: SiLU(acc1) * acc2
    silu_acc1 = acc1 * tl.sigmoid(acc1)
    result = silu_acc1 * acc2

    # Single write to output
    tl.store(C_ptr + offsets, result.to(tl.float16), mask=mask)
```

---

## Expected Impact

| Metric | Current | Expected | Reasoning |
|--------|---------|----------|-----------|
| A matrix loads | 2× | 1× | Tile reuse in K-loop |
| Intermediate DRAM | 16.8 MB | 0 | In-register computation |
| Kernel launches | 3 | 1 | Single fused kernel |
| **Latency** | 30 μs | **15-18 μs** | 40-50% memory reduction |

---

## Implementation Complexity: **High**

- Need to implement FP4 block-scaled loads in Triton
- Scale factor permutation logic is complex
- Must handle 128×128 tile size constraint
- Auto-tuning may require many iterations

**Estimated Time**: 8-12 hours

---

## Risk Level: **Medium-High**

**Risks**:
1. Triton FP4 support may have bugs on Blackwell (SM100)
2. Block scaling layout may not match cuBLAS expectations
3. Performance may be worse than cuBLAS if auto-tuning fails

**Mitigations**:
1. Start with FP16 prototype, add FP4 after correctness verified
2. Reuse existing scale factor transformation code
3. Manual tuning as fallback

---

## Evidence/Precedent

- Triton block-scaled matmul tutorial shows FP4 support: [triton-lang.org/tutorials/10-block-scaled-matmul]
- Flash Attention uses similar dual-output pattern (Q×K^T, then ×V)
- Memory bandwidth analysis is standard GPU performance modeling

---

## Rollback Plan

If Triton kernel fails:
1. Keep submission_best.py as default
2. Triton kernel in separate file, not committed to main
3. Can abandon without affecting existing code

---

*"The mathematics don't lie. Memory traffic is the bottleneck, and fusion is the solution."*

— Dr. Chen
