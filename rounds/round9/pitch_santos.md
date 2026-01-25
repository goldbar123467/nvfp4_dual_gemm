# PITCH B: Persistent Kernel with Epilogue Fusion

**Contestant**: Dr. Santos (Postdoc)

---

## The Practitioner's View

I've shipped three production ML systems. Here's what I know: **fancy theory often loses to practical engineering**.

Our current CUDA Graphs approach is actually solid. The problem isn't the three kernels — it's that we're not using the GPU efficiently.

### The Real Bottleneck

Looking at the numbers:
- Best: 30 μs (CUDA Graphs)
- Target: 13 μs (fused)
- SOL: 4.7-8.7 μs

The 30→13μs gap is suspicious. Let me break it down:

```
Kernel launch overhead (eliminated by graphs): ~15μs → 0μs ✓
Remaining time: ~30μs

Two FP4 GEMMs at peak efficiency:
  Each GEMM ≈ 512 × 4096 × 7168 × 2 = 30.1 GFLOPS
  B200 FP4 peak: 18 PFLOPS
  Theoretical time per GEMM: 30.1G / 18P = 1.67 μs
  Two GEMMs: 3.34 μs

We're at 30μs for 3.34μs of compute = ~10% efficiency!
```

**The GEMMs aren't running efficiently. This isn't a memory problem, it's a utilization problem.**

---

## Proposed Solution: Persistent Kernel + Fused Epilogue

### Step 1: Fuse the Epilogue (Easy Win)

Instead of three graph nodes, make it two:
```python
# Current: GEMM1 → GEMM2 → SiLU_mul
# Proposed: GEMM1 → GEMM2_with_fused_epilogue

# GEMM2 kernel already reads R1 to apply epilogue
# Just add SiLU and multiply in the same read pass
```

This is achievable with PyTorch custom autograd:

```python
class FusedSiLUMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gemm1_out, gemm2_out):
        # CUDA kernel that reads both, applies silu(x)*y
        return _fused_silu_mul_cuda(gemm1_out, gemm2_out)
```

**Expected gain**: 2-5 μs (eliminates R1 read in epilogue kernel)

### Step 2: Persistent GEMM (Medium Win)

The real problem: wave quantization.

For M=512, N=4096 with 128×128 tiles:
```
Tiles in M: 512/128 = 4
Tiles in N: 4096/128 = 32
Total tiles: 4 × 32 = 128 tiles

B200 has 168 SMs
128 tiles < 168 SMs → 40 SMs idle!
```

**Solution**: Persistent kernel that keeps all SMs busy across the K-dimension.

```python
# Pseudo-code for persistent GEMM
def persistent_gemm():
    while tiles_remaining:
        tile = atomic_fetch_next_tile()
        for k_chunk in tile.k_chunks:
            partial_sum = mma(A_chunk, B_chunk)
            atomic_add_to_output(partial_sum)
```

**Expected gain**: 5-10 μs (better SM utilization)

---

## Expected Impact

| Optimization | Effort | Expected Gain | Cumulative |
|-------------|--------|---------------|------------|
| Baseline (graphs) | Done | - | 30 μs |
| Fused SiLU×mul epilogue | 2 hours | 2-5 μs | 25-28 μs |
| Persistent GEMM | 6 hours | 5-10 μs | 18-23 μs |

**Total Expected**: 18-23 μs

---

## Implementation Complexity: **Medium**

- Fused epilogue: Straightforward CUDA kernel
- Persistent GEMM: More complex, but well-documented patterns exist

**Estimated Time**: 6-8 hours total

---

## Risk Level: **Low-Medium**

**Risks**:
1. Fused epilogue might not auto-vectorize well
2. Persistent GEMM atomics could cause contention

**Mitigations**:
1. Start with fused epilogue only (easy win, low risk)
2. Test persistent GEMM separately before integrating

---

## Evidence/Precedent

- CUTLASS has persistent GEMM variants (Stream-K)
- Fused epilogues are standard in production GEMM libraries
- FlashAttention uses similar persistent patterns

---

## Rollback Plan

Each optimization is independent:
1. Fused epilogue fails → keep separate kernel
2. Persistent GEMM fails → keep tiled approach
3. Both fail → current CUDA graphs still works

---

## Why This Beats the Competition

**vs. Dr. Chen (Triton)**:
- Triton FP4 is bleeding edge, likely buggy
- My approach uses proven cuBLAS for GEMMs
- Lower risk, faster implementation

**vs. Dr. Kim (Conservative)**:
- Her approach won't close the gap
- We need to take *some* risk to improve

**vs. Dr. Okonkwo (Wild Card)**:
- Cross-domain ideas are great for papers
- We need working code by end of day

---

*"I've shipped code that runs in production. Trust the patterns that work."*

— Dr. Santos
