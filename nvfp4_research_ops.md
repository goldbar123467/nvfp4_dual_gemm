# NVFP4 Dual GEMM Research Ops Sheet
## Target: B200 Blackwell, FP4 Block-Scaled GEMM with SiLU Fusion

---

## CRITICAL NUMBERS (Speed of Light Analysis)

| M | N | K | L | SOL Target [us] | Memory-Bound? |
|---|---|---|---|-----------------|---------------|
| 256 | 4096 | 7168 | 1 | 4.708 | Check FLOPS vs BW |
| 512 | 4096 | 7168 | 1 | 8.714 | |
| 256 | 3072 | 4096 | 1 | 2.125 | |
| 512 | 3072 | 7168 | 1 | 6.535 | |

### B200 Hardware Specs (per GPU)
- **FP4 Tensor Core**: 18 PFLOPS (144 PFLOPS / 8 GPUs)
- **FP8 Tensor Core**: 9 PFLOPS (72 PFLOPS / 8 GPUs)
- **HBM3e Bandwidth**: ~8 TB/s
- **Memory**: 180 GB HBM3e

### FLOPS Calculation for Your Problem
```
DUAL_GEMM_FLOPS = 2 * (2 * M * N * K)  # Two GEMMs
For M=512, N=4096, K=7168:
FLOPS = 2 * 2 * 512 * 4096 * 7168 = ~60.1 GFLOPS per operation
```

---

## TRACK 1: CUTLASS/CuTe Documentation

### PRIMARY TARGETS (fetch these URLs)

| Resource | URL | Priority |
|----------|-----|----------|
| **NVFP4 GEMM Example 72** | https://github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu | CRITICAL |
| **CUTLASS 3.x GEMM API** | https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api_3x.html | CRITICAL |
| **Colfax Sub-byte GEMM Tutorial** | https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/ | CRITICAL |
| **Blackwell Functionality MD** | https://github.com/NVIDIA/cutlass/blob/main/media/docs/blackwell_functionality.md | HIGH |
| **Example 49 Collective Builder** | https://github.com/NVIDIA/cutlass/blob/main/examples/49_hopper_gemm_with_collective_builder/49_collective_builder.cu | HIGH |
| **CUTLASS Quickstart (Blackwell)** | https://docs.nvidia.com/cutlass/latest/media/docs/cpp/quickstart.html | HIGH |

### Key Code Patterns to Extract

```cpp
// From Example 72 - NVFP4 Element Types
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementSFA = cutlass::float_e4m3fnuz_t;  // Scale factors
using MmaTileShape = Shape<_128,_128,_256>;
using ClusterShape = Shape<_1,_1,_1>;
constexpr int InputSFVectorSize = 16;  // Block size for scales

// Architecture tags
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
```

---

## TRACK 2: cuBLAS Block Scaling Documentation

### PRIMARY TARGETS

| Resource | URL | Priority |
|----------|-----|----------|
| **cuBLAS 1D Block Scaling Layout** | https://docs.nvidia.com/cuda/cublas/contents.html (Section 3.1.4.4) | CRITICAL |
| **cuBLAS Blog (12.9 FP4)** | https://developer.nvidia.com/blog/boosting-matrix-multiplication-speed-and-flexibility-with-nvidia-cublas-12-9 | HIGH |
| **Triton Block Scaled Matmul** | https://triton-lang.org/main/getting-started/tutorials/10-block-scaled-matmul.html | REFERENCE |

### Scale Factor Layout Understanding

From your task's `to_blocked` function, the scale factor layout is:
```python
# Input: (M, K//16) in row-major
# Output: Blocked format for hardware

# Atom shapes for scale factors:
atom_m = (32, 4)  # 128 rows per block
atom_k = 4

# Reorder pattern: (32, 4, rest_m, 4, rest_k, l)
# The permutation: (3, 4, 1, 5, 2, 0)
```

This matches the cuBLAS 1D block scaling layout where:
- 16-element vectors along K
- Scale factors interleaved in 128x4 tiles

---

## TRACK 3: Epilogue Fusion (SiLU) Patterns

### PRIMARY TARGETS

| Resource | URL | Priority |
|----------|-----|----------|
| **EVT Colfax Tutorial** | https://research.colfax-intl.com/epilogue_visitor_tree/ | CRITICAL |
| **CUTLASS 3.x Blog (Epilogue)** | https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/ | HIGH |
| **CUTLASS CHANGELOG (SiLU)** | https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md | REFERENCE |

### SiLU Activation Pattern
```cpp
// SiLU(x) = x * sigmoid(x)
// Your fusion: C = SiLU(A @ B1) * (A @ B2)

// CUTLASS has built-in SiLU (from CHANGELOG: "Silu, Hardswish, Leaky Relu")
// Located in: cutlass/epilogue/thread/activation.h

template<typename T>
struct SiLU {
  CUTLASS_HOST_DEVICE
  T operator()(T x) const {
    return x * sigmoid(x);
  }
};
```

### Dual GEMM Fusion Strategy

**Option A: Two Separate Kernels (Naive)**
- GEMM1: result1 = A @ B1
- GEMM2: result2 = A @ B2
- Fused epilogue: C = SiLU(result1) * result2
- Problem: A loaded twice from DRAM

**Option B: Fused Dual GEMM (Optimal)**
```cpp
// Load A tile once
// Compute both A@B1 and A@B2 in same kernel
// Fuse SiLU + multiply in epilogue

// Custom EVT for dual output:
using Fusion = cutlass::epilogue::fusion::Sm100LinearCombination<
  ElementD,     // Output type
  ElementCompute,
  ElementC,
  ElementScalar,
  /* custom SiLU multiply */
>;
```

---

## TRACK 4: Optimization Patterns

### PRIMARY TARGETS

| Resource | URL | Priority |
|----------|-----|----------|
| **Road to Petaflop Blog** | https://www.jackyoustra.com/blog/road-to-petaflop | HIGH |
| **CUTLASS Profiler** | Use `--use-cuda-graphs` flag | USEFUL |

### Key Optimization Checklist

- [ ] **TMA (Tensor Memory Accelerator)**: Use async copies for A, B, scale factors
- [ ] **Warp Specialization**: Producer/consumer pattern for Blackwell
- [ ] **TMEM (Tensor Memory)**: New Blackwell memory type for intermediates
- [ ] **Cluster Launch Control (CLC)**: Dynamic scheduling
- [ ] **Double Buffering**: Accumulator double-buffering in TMEM
- [ ] **Shared Memory Swizzling**: Use `sm100_smem_selector` for optimal layout

### Memory Access Pattern
```
// From task description:
// a: M x K x L in K-major (row-major effectively)
// b1, b2: N x K x L in K-major
// sfa, sfb1, sfb2: M x (K//16) x L or N x (K//16) x L

// For FP4: each element is 4 bits, packed 2 per byte
// Alignment: 32 elements = 16 bytes (minimum for TMA)
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
```

---

## IMPLEMENTATION ROADMAP

### Phase 1: Get Single FP4 GEMM Working (4 hours)
1. Clone CUTLASS, build example 72
2. Verify scale factor layout matches your `create_scale_factor_tensors`
3. Get single GEMM timing baseline

### Phase 2: Dual GEMM Structure (4 hours)
1. Modify mainloop to compute both B1 and B2
2. Store both accumulators
3. Basic epilogue (no fusion yet)

### Phase 3: SiLU Fusion (4 hours)
1. Add custom EVT for SiLU
2. Fuse multiply with second accumulator
3. Output to fp16

### Phase 4: Optimization (Remaining time)
1. Profile with nsight-compute
2. Tune tile sizes for your problem sizes
3. Memory coalescing optimization

---

## SILU FUSION EVT PATTERN (From Colfax Tutorial)

### EVT Structure for SiLU(A@B1) * (A@B2)

```cpp
// SiLU functor (already in CUTLASS: cutlass/epilogue/thread/activation.h)
template<typename T>
struct SiLU {
  CUTLASS_HOST_DEVICE
  T operator()(T x) const {
    return x * T(1) / (T(1) + exp(-x));  // x * sigmoid(x)
  }
};

// Custom EVT for dual accumulator fusion
// Assuming acc1 = A@B1, acc2 = A@B2
// Output = SiLU(acc1) * acc2

using DualGemmSiLUFusion =
  Sm90EVT<
    Sm90Compute<cutlass::multiplies, ElementOutput, ElementCompute, RoundStyle>,
    // Child 1: SiLU(acc1)
    Sm90EVT<
      Sm90Compute<cutlass::epilogue::thread::SiLU, ElementCompute, ElementCompute, RoundStyle>,
      Sm90AccFetch  // First accumulator (A @ B1)
    >,
    // Child 2: acc2 - Need custom fetch for second accumulator!
    Sm90Acc2Fetch  // You'll need to implement this or store acc2 differently
  >;
```

### Challenge: Dual Accumulator Access

The tricky part is that standard CUTLASS epilogue expects ONE accumulator. For dual GEMM:

**Option A: Sequential GEMMs with fusion**
```cpp
// GEMM 1: Compute A @ B1, store intermediate
// GEMM 2: Compute A @ B2, load intermediate, fuse SiLU + multiply
```

**Option B: Custom mainloop with dual accumulator**
```cpp
// Modify CollectiveMainloop to compute both mma(A, B1) and mma(A, B2)
// Store both accumulators in TMEM
// Custom epilogue reads both
```

**Option C: Use AuxLoad for first result**
```cpp
using DualGemmSiLU =
  Sm90EVT<
    Sm90Compute<multiplies, ElementOutput, ElementCompute, RoundStyle>,
    // Load pre-computed SiLU(A@B1) from memory
    Sm90EVT<
      Sm90Compute<SiLU, ElementCompute, ElementCompute, RoundStyle>,
      Sm90AuxLoad<>  // Load A@B1 result
    >,
    // Current accumulator is A@B2
    Sm90AccFetch
  >;
```

### Recommended Starting Approach

1. **Phase 1**: Get two separate GEMMs working with correct outputs
2. **Phase 2**: Fuse second GEMM's epilogue to load first result + apply SiLU + multiply
3. **Phase 3**: Optimize by fusing A loads if time permits

---

## QUICK START COMMANDS

```bash
# Clone CUTLASS
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass

# Build for Blackwell (SM100)
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=100 \
         -DCUTLASS_ENABLE_EXAMPLES=ON \
         -DCUTLASS_LIBRARY_KERNELS="cutlass3x*"
make -j$(nproc)

# Run FP4 example
./examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm --m=512 --n=4096 --k=7168
```

---

## GOTCHAS TO WATCH

1. **Scale Factor Alignment**: Must be 128-row aligned (your `atom_m = (32, 4)` = 128)
2. **K Divisibility**: K must be divisible by 256 (stated in task)
3. **FP4 Packing**: Two FP4 values per byte, use `float4_e2m1fn_x2`
4. **Scale Factor Type**: `float8_e4m3fnuz` (unsigned zero variant)
5. **Output Type**: fp16, but accumulator is fp32
