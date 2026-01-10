# NVFP4 Block Scaled Dual GEMM with SiLU Activation

## Project Overview

High-performance CUDA kernel implementation for NVIDIA B200 GPU computing:
```
C = silu(A @ B1) * (A @ B2)
```

## Data Types & Layout

| Tensor | Shape | Format | Data Type |
|--------|-------|--------|-----------|
| a | M x K x L | K-major | nvfp4 (e2m1) |
| b1, b2 | N x K x L | K-major | nvfp4 (e2m1) |
| sfa | M x (K/16) x L | K-major | fp8 (e4m3fnuz) |
| sfb1, sfb2 | N x (K/16) x L | K-major | fp8 (e4m3fnuz) |
| c | M x N x L | - | fp16 |

## Constraints

- M divisible by mma_tiler_mn[0]
- N divisible by mma_tiler_mn[1]
- K divisible by 256
- Scale factor block size: 16

## Speed of Light Targets (B200 @ 1.5GHz)

| M | N | K | L | Target (us) |
|---|---|---|---|-------------|
| 256 | 4096 | 7168 | 1 | 4.708 |
| 512 | 4096 | 7168 | 1 | 8.714 |
| 256 | 3072 | 4096 | 1 | 2.125 |
| 512 | 3072 | 7168 | 1 | 6.535 |

## Implementation Guidelines

### Performance Priorities
1. Maximize FP4 Tensor Core utilization
2. Optimize DRAM memory throughput
3. Minimize data movement between shared memory and registers
4. Fuse silu activation and element-wise multiply

### Key Optimizations to Consider
- Block scaling factor layout matches cuBLAS format
- Dual GEMM fusion to reuse A matrix loads
- Shared memory tiling for optimal cache reuse
- Warp-level MMA instructions for FP4
- Asynchronous memory copies (cp.async)

### Scale Factor Layout
Uses cuBLAS block scaling format:
- Atom shape: (32, 4) for M/N dimension, 4 for K dimension
- Permute order: (32, 4, rest_m, 4, rest_k, l)
- Reference: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

## Validation

- rtol: 1e-03
- atol: 1e-03
- Ranking: geometric mean of benchmark results

## File Structure

```
nvfp4-dual-gemm/
├── CLAUDE.md           # This file
├── task.md             # Original task specification
├── src/
│   ├── kernel.cu       # Main CUDA kernel
│   ├── kernel.cuh      # Kernel headers
│   └── utils.cuh       # Helper functions
├── python/
│   ├── task.py         # Python bindings
│   └── benchmark.py    # Benchmarking utilities
└── tests/
    └── test_kernel.py  # Validation tests
```

## Development Commands

```bash
# Build
nvcc -arch=sm_100 -O3 src/kernel.cu -o kernel

# Test
python tests/test_kernel.py

# Benchmark
python python/benchmark.py
```

## Notes

- Target GPU: NVIDIA B200 (SM 100, Blackwell architecture)
- FP4 (e2m1) values: [-1.5, -1, -0.5, 0, +0.5, +1, +1.5]
- Uses torch.float4_e2m1fn_x2 packed format (2 FP4 values per byte)
