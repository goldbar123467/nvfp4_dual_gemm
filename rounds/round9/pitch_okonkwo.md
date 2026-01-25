# PITCH D: Dual-Accumulator CUTLASS Kernel with EVT Fusion

**Contestant**: Dr. Okonkwo (Visiting Researcher)

---

## Fresh Eyes on an Old Problem

I just arrived from Prof. Chen's lab at Berkeley where we worked on Flash Attention. Looking at this problem, I see a familiar pattern that everyone here has missed.

**This is not a GEMM optimization problem. This is a memory-bound fused kernel problem.**

---

## The Insight From Flash Attention

In Flash Attention, we faced a similar challenge:
- Multiple matrix operations (Q×K^T, softmax, ×V)
- Intermediate results too large for SRAM
- Solution: **Tile across ALL operations simultaneously**

Your dual-GEMM has the same structure:
```
Flash Attention:    O = softmax(Q × K^T) × V
Your Problem:       C = SiLU(A × B1) × (A × B2)
```

Both have:
- Shared input matrix (Q/A)
- Two dependent operations
- Element-wise fusion in epilogue

---

## Proposed Solution: CUTLASS Dual-Accumulator Kernel

### The Key Trick: Dual Accumulator Registers

CUTLASS lets us hold **two separate accumulators** in the same kernel:

```cpp
// Standard GEMM mainloop has ONE accumulator
AccumulatorType acc;
for (int k = 0; k < K; k += TILE_K) {
    acc += mma(A_tile, B_tile);
}

// OUR kernel has TWO accumulators
AccumulatorType acc1, acc2;  // Both in registers
for (int k = 0; k < K; k += TILE_K) {
    // Load A tile ONCE
    auto A_tile = load_fp4_scaled(A, scale_A, k);

    // Load B1 tile, accumulate GEMM1
    auto B1_tile = load_fp4_scaled(B1, scale_B1, k);
    acc1 += mma(A_tile, B1_tile);

    // Load B2 tile, accumulate GEMM2 (A_tile reused!)
    auto B2_tile = load_fp4_scaled(B2, scale_B2, k);
    acc2 += mma(A_tile, B2_tile);
}

// Fused epilogue - ALL IN REGISTERS
auto result = silu(acc1) * acc2;
store(C, result);
```

### Register Pressure Analysis

For 128×128 tile with FP32 accumulators:
```
Per accumulator: 128 × 128 × 4 bytes = 64 KB
Two accumulators: 128 KB
B200 SM has: 256 KB registers

We fit! (with room for A, B1, B2 tiles)
```

---

## CUTLASS Implementation Path

### Step 1: Fork Example 72 (2 hours)

Start from `72b_blackwell_nvfp4_nvfp4_gemm.cu` which already handles:
- FP4 block-scaled loads
- Scale factor permutation
- 128×128 tile constraints

### Step 2: Modify Mainloop (4 hours)

```cpp
// In CollectiveMma.hpp, modify the mainloop
template <class TiledMma>
struct DualCollectiveMma {
    using AccType = typename TiledMma::AccumulatorType;

    CUTLASS_DEVICE
    void operator()(
        TiledMma& mma,
        FragmentA& fragA,
        FragmentB1& fragB1,
        FragmentB2& fragB2,
        AccType& acc1,
        AccType& acc2
    ) {
        // Load A (shared)
        copy(fragA, A_smem);

        // GEMM1: A × B1
        copy(fragB1, B1_smem);
        mma(acc1, fragA, fragB1, acc1);

        // GEMM2: A × B2 (fragA reused!)
        copy(fragB2, B2_smem);
        mma(acc2, fragA, fragB2, acc2);
    }
};
```

### Step 3: EVT Epilogue Fusion (2 hours)

Use CUTLASS Epilogue Visitor Tree for SiLU × multiply:

```cpp
using DualGemmEpilogue =
  cutlass::epilogue::fusion::Sm100EVT<
    // Outer: multiply
    cutlass::epilogue::fusion::Sm100Compute<
      cutlass::multiplies,
      ElementOutput,
      ElementCompute
    >,
    // Left child: SiLU(acc1)
    cutlass::epilogue::fusion::Sm100EVT<
      cutlass::epilogue::fusion::Sm100Compute<
        cutlass::epilogue::thread::SiLU,
        ElementCompute,
        ElementCompute
      >,
      cutlass::epilogue::fusion::Sm100AccFetch<0>  // First accumulator
    >,
    // Right child: acc2
    cutlass::epilogue::fusion::Sm100AccFetch<1>  // Second accumulator
  >;
```

---

## Expected Impact

| Metric | Current | Expected | Reasoning |
|--------|---------|----------|-----------|
| A loads | 2× | 1× | Tile reuse in mainloop |
| Kernel launches | 3 | 1 | Single fused kernel |
| Intermediate DRAM | 16.8 MB | 0 | Register computation |
| **Latency** | 30 μs | **12-15 μs** | Near SOL |

---

## Implementation Complexity: **High**

- Requires CUTLASS expertise
- Custom mainloop modification
- Dual accumulator EVT is novel

**Estimated Time**: 10-14 hours

---

## Risk Level: **Medium**

**Risks**:
1. Register pressure may force spilling
2. Dual accumulator pattern is untested on SM100
3. EVT with two accumulators may not be supported

**Mitigations**:
1. Profile register usage with Nsight
2. Fall back to smaller tiles if needed
3. Manual epilogue as backup

---

## Evidence/Precedent

- Flash Attention uses dual-accumulator pattern (Q×K and ×V)
- CUTLASS has `Sm90DualGemm` template (we adapt for SM100)
- EVT is designed for complex epilogue fusion

### Relevant CUTLASS Files

```
cutlass/
├── examples/
│   └── 72_blackwell_narrow_precision_gemm/  # Our starting point
├── include/cutlass/epilogue/fusion/
│   └── operations.hpp                       # EVT operations
│   └── sm90_callbacks_tma_warpspecialized.hpp  # Epilogue templates
└── include/cutlass/gemm/collective/
    └── sm100_mma_nvfp4.hpp                  # FP4 MMA for Blackwell
```

---

## Why This Is Different

**vs. Dr. Chen (Triton)**:
- CUTLASS has native FP4 support, Triton's is experimental
- CUTLASS EVT is more mature than Triton epilogue fusion
- We leverage existing example 72 code

**vs. Dr. Santos (Persistent)**:
- Persistent helps utilization, but doesn't fix memory traffic
- We fix the fundamental problem: redundant A loads

**vs. Dr. Kim (Conservative)**:
- Safe approaches won't close a 2.3× gap
- We need architectural change, not configuration tuning

---

## Rollback Plan

1. If dual accumulator fails: fall back to two separate CUTLASS GEMMs
2. If EVT fails: use manual epilogue kernel (like current)
3. If CUTLASS fails: keep CUDA Graphs approach

---

## The Cross-Domain Insight

The reason this approach isn't obvious is that:
1. GPU MODE challenge framing focuses on "GEMM optimization"
2. But this is a **fused kernel** problem like Flash Attention
3. The solution is well-known in attention literature
4. It just hasn't been applied to dual-GEMM + SiLU

**"When you have a hammer, everything looks like a nail. When you worked on Flash Attention, everything looks like tiled fusion."**

---

*"I've seen this problem before. It's Flash Attention with a different epilogue."*

— Dr. Okonkwo
