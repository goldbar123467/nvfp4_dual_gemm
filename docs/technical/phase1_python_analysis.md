# Phase 1: Python Code Analysis for CUDA Kernel Integration

## Executive Summary

This document analyzes the Python interface code in `/home/ubuntu/projects/nvfp4-dual-gemm/python/` to understand the kernel interface, data flow, and integration requirements for the CUDA kernel implementation.

**Operation**: `C = silu(A @ B1) * (A @ B2)`

---

## 1. Kernel Interface Specification

### 1.1 Input Tensors

| Tensor | Shape | Data Type | Description |
|--------|-------|-----------|-------------|
| `a` | `[M, K//2, L]` | `torch.float4_e2m1fn_x2` | Input matrix A (FP4 packed, 2 values/byte) |
| `b1` | `[N, K//2, L]` | `torch.float4_e2m1fn_x2` | Weight matrix 1 (FP4 packed) |
| `b2` | `[N, K//2, L]` | `torch.float4_e2m1fn_x2` | Weight matrix 2 (FP4 packed) |
| `sfa` | `[M, K//16, L]` | `torch.float8_e4m3fn` | Scale factors for A (reference format) |
| `sfb1` | `[N, K//16, L]` | `torch.float8_e4m3fn` | Scale factors for B1 (reference format) |
| `sfb2` | `[N, K//16, L]` | `torch.float8_e4m3fn` | Scale factors for B2 (reference format) |
| `sfa_permuted` | `[32, 4, M//128, 4, K//64, L]` | `torch.float8_e4m3fn` | Scale factors for A (kernel format) |
| `sfb1_permuted` | `[32, 4, N//128, 4, K//64, L]` | `torch.float8_e4m3fn` | Scale factors for B1 (kernel format) |
| `sfb2_permuted` | `[32, 4, N//128, 4, K//64, L]` | `torch.float8_e4m3fn` | Scale factors for B2 (kernel format) |

### 1.2 Output Tensor

| Tensor | Shape | Data Type | Description |
|--------|-------|-----------|-------------|
| `c` | `[M, N, L]` | `torch.float16` | Output matrix |

### 1.3 Internal Computation

- **Accumulator dtype**: `torch.float32` (FP32 for numerical stability)
- **GEMM1**: `A @ B1.T` -> `[M, N, L]` in FP32
- **GEMM2**: `A @ B2.T` -> `[M, N, L]` in FP32
- **Activation**: `silu(GEMM1) * GEMM2` computed in FP32
- **Output conversion**: FP32 -> FP16

---

## 2. Dimensional Constraints

From `constants.py`:

```
SCALE_FACTOR_BLOCK_SIZE = 16   # FP4 elements per scale factor
K_DIVISIBILITY = 256           # K must be divisible by 256
M_ALIGNMENT = 128              # M must be divisible by 128 (for MMA tiling)
N_ALIGNMENT = 128              # N must be divisible by 128 (for MMA tiling)

ATOM_M = (32, 4)               # 32 rows per atom, 4 atoms per 128-block
ATOM_K = 4                     # 4 scale blocks per K tile (64 FP4 elements)
```

**Derived relationships**:
- Scale factor K dimension: `sf_k = K // 16`
- Packed FP4 K dimension: `K // 2` (bytes)
- Number of M blocks: `M // 128`
- Number of K tiles: `sf_k // 4 = K // 64`

---

## 3. Scale Factor Layout Transformation

### 3.1 Reference Format (for PyTorch `_scaled_mm`)

Shape: `[M, K//16, L]` or `[N, K//16, L]`
- Simple 3D tensor with M/N rows, sf_k columns, L batches

### 3.2 Kernel Format (cuBLAS block scaling layout)

Shape: `[32, 4, rest_m, 4, rest_k, L]`

Where:
- `rest_m = M // 128` (number of 128-row blocks)
- `rest_k = K // 64` (number of 64-element K tiles)

### 3.3 Index Mapping

The transformation maps from reference `[i, j, b]` to permuted `[mm32, mm4, mm, kk4, kk, b]`:

```python
# atom_m = (32, 4), atom_k = 4

mm   = i // 128           # Which 128-row block
mm32 = i % 32             # Position within 32-row atom
mm4  = (i % 128) // 32    # Which 32-row atom within 128-block (0-3)
kk   = j // 4             # Which K tile (groups of 4 scale factors)
kk4  = j % 4              # Position within K tile (0-3)
```

### 3.4 Visualization

```
Reference Layout [M, K//16, L]:
+------------------------------------------+
|  Row 0:   sf[0,0] sf[0,1] sf[0,2] ...    |
|  Row 1:   sf[1,0] sf[1,1] sf[1,2] ...    |
|  ...                                      |
|  Row 127: sf[127,0] sf[127,1] ...        |  <- First 128-row block
+------------------------------------------+
|  Row 128: sf[128,0] sf[128,1] ...        |  <- Second 128-row block
|  ...                                      |
+------------------------------------------+

Kernel Layout [32, 4, rest_m, 4, rest_k, L]:
+--------------------------------------------------+
| Atom[0..31, 0, block_m, 0..3, tile_k, batch]    |
|   32 rows x 4 positions x blocks x 4 k-pos x ... |
+--------------------------------------------------+
```

---

## 4. FP4 Packed Format

### 4.1 Bit Layout

`torch.float4_e2m1fn_x2` packs 2 FP4 values per byte:
- **Low nibble (bits 0-3)**: First FP4 value
- **High nibble (bits 4-7)**: Second FP4 value

### 4.2 Valid FP4 Values

Each nibble uses format e2m1 (2 exponent bits, 1 mantissa bit):
```
Valid values: [-1.5, -1.0, -0.5, 0.0, +0.5, +1.0, +1.5]
```

### 4.3 Generation Mask

```python
ref_i8 = ref_i8 & 0b1011_1011  # Keep sign bit and 2 LSBs per nibble
```

---

## 5. Data Flow Diagram

```
                              INPUT GENERATION
                              ================

  +-----------+    +-----------+    +-----------+
  |  Random   |    |  Random   |    |  Random   |
  |  uint8    |    |  uint8    |    |  uint8    |
  +-----------+    +-----------+    +-----------+
       |               |               |
       v               v               v
  [mask 0b1011_1011]   ...            ...
       |               |               |
       v               v               v
  +-----------+    +-----------+    +-----------+
  |    A      |    |    B1     |    |    B2     |
  | [M,K/2,L] |    | [N,K/2,L] |    | [N,K/2,L] |
  |   FP4x2   |    |   FP4x2   |    |   FP4x2   |
  +-----------+    +-----------+    +-----------+

                    SCALE FACTORS
                    =============

  +----------------+     +----------------+     +----------------+
  |      SFA       |     |      SFB1      |     |      SFB2      |
  | [M, K/16, L]   |     | [N, K/16, L]   |     | [N, K/16, L]   |
  |      FP8       |     |      FP8       |     |      FP8       |
  +----------------+     +----------------+     +----------------+
         |                     |                     |
         v                     v                     v
  +------------------+  +------------------+  +------------------+
  |  SFA_permuted    |  |  SFB1_permuted   |  |  SFB2_permuted   |
  |[32,4,M/128,4,    |  |[32,4,N/128,4,    |  |[32,4,N/128,4,    |
  |    K/64,L]       |  |    K/64,L]       |  |    K/64,L]       |
  +------------------+  +------------------+  +------------------+


                        KERNEL EXECUTION
                        ================

  +-------+   +--------+   +--------+   +---------------+   +----------------+
  |   A   |   |   B1   |   |   B2   |   | SFA_permuted  |   | SFB1_permuted  |
  +-------+   +--------+   +--------+   +---------------+   +----------------+
      |           |            |              |                    |
      +-----+-----+            |              +--------+-----------+
            |                  |                       |
            v                  v                       v
      +----------+       +----------+           +--------------+
      |  GEMM1   |       |  GEMM2   |           | Block Scales |
      | A @ B1.T |       | A @ B2.T |           |   Applied    |
      +----------+       +----------+           +--------------+
            |                  |
            v                  v
      +-----------+      +-----------+
      |    FP32   |      |    FP32   |
      | [M, N, L] |      | [M, N, L] |
      +-----------+      +-----------+
            |                  |
            v                  |
      +----------+             |
      |   SiLU   |             |
      | x*sig(x) |             |
      +----------+             |
            |                  |
            +--------+---------+
                     |
                     v
             +---------------+
             |   Multiply    |
             | silu(G1) * G2 |
             +---------------+
                     |
                     v
             +---------------+
             |    Output     |
             | [M, N, L] FP16|
             +---------------+
```

---

## 6. Reference Implementation Logic

### 6.1 Algorithm Steps

1. **Input validation**: Check all dtypes and contiguity
2. **Allocate FP32 accumulators**: `ref1, ref2` shape `[M, N, L]`
3. **Per-batch loop** (for each `l_idx` in `0..L`):
   - Convert scale factors to blocked format using `to_blocked()`
   - Execute `_scaled_mm(a[:,:,l], b1[:,:,l].T, scale_a, scale_b1)` -> `ref1[:,:,l]`
   - Execute `_scaled_mm(a[:,:,l], b2[:,:,l].T, scale_a, scale_b2)` -> `ref2[:,:,l]`
4. **Fused activation**: `c = silu(ref1) * ref2`
5. **Type conversion**: FP32 -> FP16

### 6.2 SiLU Activation

```python
silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
```

### 6.3 `to_blocked()` Function for `_scaled_mm`

Converts `[rows, cols]` scale factors to cuBLAS blocked format:

```python
def to_blocked(input_matrix):
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    blocks = input_matrix.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()
```

---

## 7. CUDA Kernel Integration Plan

### 7.1 Kernel Signature (Proposed)

```cpp
void nvfp4_dual_gemm_silu(
    // FP4 packed inputs
    const uint8_t* a,           // [M, K/2, L] - FP4x2 packed
    const uint8_t* b1,          // [N, K/2, L] - FP4x2 packed
    const uint8_t* b2,          // [N, K/2, L] - FP4x2 packed

    // Scale factors (kernel format - permuted)
    const fp8_e4m3* sfa,        // [32, 4, M/128, 4, K/64, L]
    const fp8_e4m3* sfb1,       // [32, 4, N/128, 4, K/64, L]
    const fp8_e4m3* sfb2,       // [32, 4, N/128, 4, K/64, L]

    // Output
    half* c,                    // [M, N, L]

    // Dimensions
    int M, int N, int K, int L
);
```

### 7.2 Python Binding Interface

```python
def custom_kernel(data: input_t) -> output_t:
    """
    Custom CUDA kernel wrapper.

    Args:
        data: Tuple from generate_input()
              (a, b1, b2, sfa, sfb1, sfb2,
               sfa_perm, sfb1_perm, sfb2_perm, c)

    Returns:
        c: Output tensor [M, N, L] in FP16
    """
    a, b1, b2, _, _, _, sfa_perm, sfb1_perm, sfb2_perm, c = data

    # Extract dimensions
    m, n, l = c.shape
    k = a.shape[1] * 2  # K/2 packed -> K

    # Call CUDA kernel (via pybind11 or torch extension)
    cuda_module.nvfp4_dual_gemm_silu(
        a.data_ptr(), b1.data_ptr(), b2.data_ptr(),
        sfa_perm.data_ptr(), sfb1_perm.data_ptr(), sfb2_perm.data_ptr(),
        c.data_ptr(),
        m, n, k, l
    )

    return c
```

### 7.3 Integration Approaches

**Option A: PyTorch C++ Extension**
- Use `torch.utils.cpp_extension.load_inline()` or `setup.py`
- Direct access to tensor data pointers
- Automatic GPU memory management

**Option B: pybind11 with CUDA**
- Standalone shared library
- More flexibility, more boilerplate
- Better for complex build configurations

**Option C: Triton Kernel**
- Python-native GPU kernel
- Easier prototyping
- May have performance limitations for FP4

### 7.4 Validation Approach

```python
# Use the provided check_implementation function
from task import check_implementation, generate_input

# Generate test inputs
inputs = generate_input(m=256, n=4096, k=7168, l=1, seed=42)

# Validate custom kernel
passed, error_info = check_implementation(custom_kernel, inputs)

if passed:
    print("Validation PASSED")
else:
    print(f"Validation FAILED: {error_info}")
```

### 7.5 Performance Targets

| M | N | K | L | Target (us) | FLOPs | AI (ops/byte) |
|---|---|---|---|-------------|-------|---------------|
| 256 | 4096 | 7168 | 1 | 4.708 | 15.1G | ~850 |
| 512 | 4096 | 7168 | 1 | 8.714 | 30.1G | ~950 |
| 256 | 3072 | 4096 | 1 | 2.125 | 6.4G | ~800 |
| 512 | 3072 | 7168 | 1 | 6.535 | 22.6G | ~900 |

---

## 8. Environment Status

**Python Environment Check**:
- Python version: 3.10.12
- PyTorch: Not installed in system Python
- CUDA/nvidia-smi: Not available on current node

**Note**: A CUDA-capable environment with PyTorch (supporting FP4/FP8) is required for testing. The target GPU is NVIDIA B200 (SM 100, Blackwell architecture).

---

## 9. Key Implementation Considerations

### 9.1 Memory Access Patterns

- **A matrix reuse**: A is used for both GEMMs - optimize for shared memory caching
- **Scale factor locality**: Permuted layout aligns with MMA tile access patterns
- **Output coalescing**: Ensure coalesced writes to C

### 9.2 Fusion Opportunities

1. Dual GEMM fusion: Load A once, compute both A@B1 and A@B2
2. Activation fusion: SiLU and multiply computed in registers before store
3. Scale factor fusion: Apply block scaling during MMA accumulation

### 9.3 Numerical Considerations

- Accumulate in FP32 for stability
- SiLU computed in FP32 before final FP16 conversion
- Tolerances: rtol=1e-3, atol=1e-3

---

## 10. Files Analyzed

| File | Purpose |
|------|---------|
| `/home/ubuntu/projects/nvfp4-dual-gemm/python/task.py` | Input generation, reference kernel, validation |
| `/home/ubuntu/projects/nvfp4-dual-gemm/python/constants.py` | Data types, constraints, assertion helpers |
| `/home/ubuntu/projects/nvfp4-dual-gemm/python/utils.py` | Utilities, FLOP calculation, validation factory |
| `/home/ubuntu/projects/nvfp4-dual-gemm/task.md` | Original task specification |

---

*Document generated: Phase 1 Python Analysis*
*Project: NVFP4 Block Scaled Dual GEMM with SiLU Activation*
