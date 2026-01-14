# Gap 1: B Matrix Layout and Transpose Contract

## Problem Statement

The specification defines B matrices (b1/b2) with the following properties:
- Shape: N x K x L (where L is batch dimension)
- Storage: K-major (K is contiguous in memory)
- Mathematical operation: C = A @ B^T where A is (M,K) and B^T is (K,N)

**The Critical Insight**: The B matrix is stored as [N, K] with K contiguous, but we need to compute the GEMM as if B were transposed to [K, N]. This is NOT an actual memory transpose - it's a layout interpretation.

### Memory Layout Breakdown

```
B stored in memory as [N, K] K-major:
  - stride_k = 1 (K dimension is contiguous)
  - stride_n = K (N dimension has stride K)
  
Physical layout for B[2,3] (N=2, K=3):
  Memory: [b00, b01, b02, b10, b11, b12]
  Logical:
    B[0,:] = [b00, b01, b02]  <- row 0 (n=0)
    B[1,:] = [b10, b11, b12]  <- row 1 (n=1)
```

### What We Need vs What We Have

| Aspect | What We Have | What GEMM Needs |
|--------|--------------|-----------------|
| B shape | [N, K] | B^T as [K, N] |
| Contiguous dim | K | K (column of B^T) |
| Math | B[n, k] | B^T[k, n] = B[n, k] |

The key realization: B[n,k] and B^T[k,n] reference the SAME memory location. No data movement is needed - only the correct layout tag interpretation.

## CUTLASS Layout Tags

CUTLASS uses layout tags to describe how matrix elements map to memory:

### `cutlass::layout::RowMajor`
- Leading dimension is the column count
- Row index changes slower than column index
- For matrix [M, N]: element (m, n) at offset `m * N + n`

### `cutlass::layout::ColumnMajor`  
- Leading dimension is the row count
- Column index changes slower than row index
- For matrix [M, N]: element (m, n) at offset `n * M + m`

### For B stored as [N,K] K-contiguous, treated as B^T in GEMM:

**Option 1: Use ColumnMajor for B**
```cpp
using LayoutB = cutlass::layout::ColumnMajor;
// Interprets B as column-major [K, N] where K is leading dim
// Element B^T[k, n] = B[n, k] at offset: n * K + k
// This matches our K-contiguous [N, K] storage!
```

**Option 2: Use RowMajor with transposed B operand**
```cpp
using LayoutB = cutlass::layout::RowMajor;
// Then set TransformB = cutlass::ComplexTransform::kConjugate or similar
// More complex, less recommended for this use case
```

## Correct Configuration

```cpp
// B is [N, K] with K contiguous (stride_n = K, stride_k = 1)
// To compute C = A @ B^T where B^T is [K, N]:

// CUTLASS 3.x style:
using LayoutB = cutlass::layout::ColumnMajor;  // K is leading dimension

// This makes CUTLASS interpret B[n,k] as B^T[k,n] without memory movement
// The MMA iterators will read:
//   - K elements contiguously (good for memory coalescing)
//   - N elements with stride K

// Stride specification:
// cute::Stride<Int<K>, Int<1>> for B shape [N, K]
// or cute::Stride<int64_t, Int<1>> for dynamic N

// Full CollectiveMainloop setup:
using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
    DispatchPolicy,
    TileShape,
    ElementA, StrideA,   // A: [M, K] row-major  
    ElementB, StrideB,   // B: [N, K] -> treated as B^T[K, N] via ColumnMajor
    TiledMma,
    GmemTiledCopyA, SmemLayoutA, SmemCopyAtomA, TransformA,
    GmemTiledCopyB, SmemLayoutB, SmemCopyAtomB, TransformB
>;
```

## Debug Kernel Gate

This is a MANDATORY correctness gate before proceeding with the full implementation.

### Test Case: Non-Symmetric 2x2 Matrices

```cpp
// Minimal 1-tile test with non-symmetric B to detect transpose errors
// Using non-symmetric B is critical - symmetric matrices hide transpose bugs!

// A = [[1, 2],    (M=2, K=2, row-major)
//      [3, 4]]

// B = [[5, 6],    (N=2, K=2, K-major storage)
//      [7, 8]]
// B stored in memory: [5, 6, 7, 8] (K contiguous)

// CORRECT: C = A @ B^T
// B^T = [[5, 7],
//        [6, 8]]
// C[0,0] = 1*5 + 2*6 = 5 + 12 = 17
// C[0,1] = 1*7 + 2*8 = 7 + 16 = 23
// C[1,0] = 3*5 + 4*6 = 15 + 24 = 39
// C[1,1] = 3*7 + 4*8 = 21 + 32 = 53
// Expected: [[17, 23], [39, 53]]

// WRONG: C = A @ B (no transpose)
// C[0,0] = 1*5 + 2*7 = 5 + 14 = 19
// C[0,1] = 1*6 + 2*8 = 6 + 16 = 22
// C[1,0] = 3*5 + 4*7 = 15 + 28 = 43
// C[1,1] = 3*6 + 4*8 = 18 + 32 = 50
// Wrong result: [[19, 22], [43, 50]]
```

### Detection Pattern

| C[0,0] | C[0,1] | C[1,0] | C[1,1] | Interpretation |
|--------|--------|--------|--------|----------------|
| 17 | 23 | 39 | 53 | CORRECT (A @ B^T) |
| 19 | 22 | 43 | 50 | WRONG (A @ B, no transpose) |
| 26 | 38 | 30 | 44 | WRONG (A^T @ B^T) |
| 23 | 31 | 34 | 46 | WRONG (A^T @ B) |

### Minimal Test Kernel

```cpp
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>

template<typename LayoutB>
__global__ void test_b_layout_kernel(
    float const* A,  // [2, 2] row-major
    float const* B,  // [2, 2] K-major (our format)
    float* C         // [2, 2] row-major output
) {
    // Simple reference implementation
    // A[m,k] at offset m*2 + k
    // B[n,k] at offset n*2 + k (K-major)
    // C[m,n] = sum_k A[m,k] * B[n,k]  (using B as B^T)
    
    int m = threadIdx.x;
    int n = threadIdx.y;
    
    if (m < 2 && n < 2) {
        float sum = 0.0f;
        for (int k = 0; k < 2; ++k) {
            float a_val = A[m * 2 + k];      // A[m, k]
            float b_val = B[n * 2 + k];      // B[n, k] = B^T[k, n]
            sum += a_val * b_val;
        }
        C[m * 2 + n] = sum;
    }
}

void validate_b_layout() {
    // Host data
    float h_A[4] = {1, 2, 3, 4};
    float h_B[4] = {5, 6, 7, 8};  // K-major: B[0,:] = {5,6}, B[1,:] = {7,8}
    float h_C[4] = {0};
    
    // Expected: [[17, 23], [39, 53]]
    float expected[4] = {17, 23, 39, 53};
    
    // Device allocations and kernel launch...
    // [implementation details]
    
    // Validation
    for (int i = 0; i < 4; ++i) {
        if (std::abs(h_C[i] - expected[i]) > 1e-5f) {
            printf("LAYOUT ERROR at C[%d]: got %f, expected %f\n", 
                   i, h_C[i], expected[i]);
            printf("This indicates B transpose is NOT being applied correctly!\n");
            exit(1);
        }
    }
    printf("B layout validation PASSED\n");
}
```

## Common Mistakes and Detection

### Mistake 1: Using RowMajor for K-contiguous B

```cpp
// WRONG:
using LayoutB = cutlass::layout::RowMajor;
// This interprets B as [N, K] with N as leading dim
// Result: computes A @ B instead of A @ B^T
```
**Detection**: C[0,0] = 19 instead of 17

### Mistake 2: Incorrect stride specification

```cpp
// WRONG:
auto stride_B = cute::make_stride(Int<1>{}, K);  // Swapped!
// This says K is contiguous in the wrong dimension
```
**Detection**: Random garbage or access violations

### Mistake 3: Double transpose

```cpp
// WRONG:
using LayoutB = cutlass::layout::ColumnMajor;
// AND
using TransformB = cutlass::ComplexTransform::kConjugate;
// Double transpose = no transpose!
```
**Detection**: C[0,0] = 19 instead of 17

### Mistake 4: Testing with symmetric B

```cpp
// DANGEROUS:
float h_B[4] = {1, 2, 2, 3};  // B = B^T for this data!
// This passes validation even with wrong layout
```
**Detection**: Use asymmetric test data like [[5,6],[7,8]]

## Validation Checklist

Before proceeding to full kernel implementation:

- [ ] Debug kernel compiles and runs
- [ ] C[0,0] == 17 (not 19)
- [ ] C[0,1] == 23 (not 22)
- [ ] C[1,0] == 39 (not 43)
- [ ] C[1,1] == 53 (not 50)
- [ ] Test with multiple B values to ensure consistency
- [ ] Verify memory access patterns with compute-sanitizer

## Integration with NVF4 Dual-GEMM

For the dual-GEMM kernel with FP4 quantized weights:

```cpp
// B1 and B2 both follow the same layout contract
// Shape: [N, K] with K contiguous
// After FP4 dequantization, use ColumnMajor interpretation

using LayoutB1 = cutlass::layout::ColumnMajor;
using LayoutB2 = cutlass::layout::ColumnMajor;

// Strides for batched operation (L batches)
// B shape: [N, K, L]
// Stride: (K, 1, N*K) in cute notation
auto make_b_stride = [](int N, int K) {
    return cute::make_stride(K, Int<1>{}, N * K);
};
```

## References

- CUTLASS Layout documentation: `include/cutlass/layout/matrix.h`
- CuTe layout algebra: `include/cute/layout.hpp`
- GEMM problem definition: `include/cutlass/gemm/gemm.h`
