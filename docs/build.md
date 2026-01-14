# NVFP4 Dual-GEMM Build Documentation

## Overview

This document provides comprehensive build instructions for the NVFP4 dual-GEMM kernel targeting NVIDIA B200 (SM100 Blackwell) architecture.

**Kernel Operation**: `C = silu(A @ B1) * (A @ B2)`

**Related Documentation:**
- [CMake Reference](build-guide/cmake-reference.md) - Detailed CMake configuration
- [Troubleshooting Guide](build-guide/troubleshooting.md) - Common errors and solutions

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [CUTLASS Setup](#cutlass-setup)
4. [Build Options](#build-options)
5. [Verification](#verification)

---

## Prerequisites

### Hardware Requirements
- NVIDIA B200 GPU (SM100 Blackwell architecture)
- Compute capability: 10.0, 10.1, or 10.3

### Software Requirements

| Component | Minimum Version | Recommended |
|-----------|-----------------|-------------|
| CUDA Toolkit | 12.8 | 12.8+ |
| GCC | 11.0 | 12.0+ |
| CMake | 3.18 | 3.24+ |
| Python | 3.8 | 3.10+ |
| PyTorch | 2.0 | 2.4+ |

### Verify Environment
```bash
# Check CUDA version (must be 12.8+)
nvcc --version

# Check GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Check PyTorch CUDA
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
```

---

## Quick Start

```bash
# 1. Setup CUTLASS
cd /workspace/nvfp4_dual_gemm
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass && mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS="100a" -DCUTLASS_ENABLE_EXAMPLES=ON -DCUTLASS_ENABLE_TESTS=OFF
make -j$(nproc) 72_blackwell_narrow_precision_gemm

# 2. Build custom extension
cd /workspace/nvfp4_dual_gemm/src
python setup.py build_ext --inplace

# 3. Verify
cd /workspace/nvfp4_dual_gemm
python -c "import sys; sys.path.insert(0, 'src'); import dual_gemm_nvfp4_cuda; print('SUCCESS')"
```

---

## CUTLASS Setup

### Clone Repository
```bash
cd /workspace/nvfp4_dual_gemm
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v3.8.0  # or latest stable
```

### Configure and Build
```bash
mkdir -p build && cd build

# Configure for SM100 Blackwell
cmake .. \
    -DCUTLASS_NVCC_ARCHS="100a" \
    -DCUTLASS_ENABLE_EXAMPLES=ON \
    -DCUTLASS_ENABLE_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release

# Build Example 72 (NVFP4 reference)
make -j$(nproc) 72_blackwell_narrow_precision_gemm
```

### Architecture Flags

| Flag | Description |
|------|-------------|
| `100a` | SM100 with all features (recommended) |
| `100f` | SM100 forward-compatible |
| `101a` | SM101 with all features |
| `103a` | SM103 with all features |

---

## Build Options

### Option A: setup.py (Recommended)

```bash
cd /workspace/nvfp4_dual_gemm/src
python setup.py build_ext --inplace
```

See [src/setup.py](../src/setup.py) for the full configuration.

### Option B: CMake

```bash
cd /workspace/nvfp4_dual_gemm/src
mkdir -p build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
make -j$(nproc)
```

See [CMake Reference](build-guide/cmake-reference.md) for detailed options.

### Option C: JIT Compilation

```python
import torch
from torch.utils.cpp_extension import load

dual_gemm_nvfp4 = load(
    name='dual_gemm_nvfp4_cuda',
    sources=['dual_gemm_nvfp4.cu', 'pytorch_bindings.cpp'],
    extra_include_paths=['../cutlass/include'],
    extra_cuda_cflags=[
        '-O3', '-arch=sm_100a', '--use_fast_math',
        '--expt-relaxed-constexpr', '-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1',
    ],
    extra_cflags=['-O3', '-std=c++17'],
    verbose=True
)
```

---

## Verification

### Test Import
```bash
python -c "import dual_gemm_nvfp4_cuda; print('SUCCESS')"
```

### Run CUTLASS Example
```bash
./cutlass/build-guide/bin/72b_blackwell_nvfp4_nvfp4_gemm --m=256 --n=4096 --k=7168
```

### Benchmark
```bash
cd /workspace/nvfp4_dual_gemm
python -c "
from python.submission import custom_kernel
from python.task import generate_input, ref_kernel
import torch

data = generate_input(256, 4096, 7168, 1, 'cuda')
result = custom_kernel(data)
print(f'Output shape: {result.shape}')
"
```

---

## Performance Targets

| M | N | K | L | SOL Target (us) |
|---|---|---|---|-----------------|
| 256 | 4096 | 7168 | 1 | 4.7 |
| 512 | 4096 | 7168 | 1 | 8.7 |
| 256 | 3072 | 4096 | 1 | 2.1 |
| 512 | 3072 | 7168 | 1 | 6.5 |

---

## Directory Structure

```
/workspace/nvfp4_dual_gemm/
├── cutlass/                      # CUTLASS library
│   ├── include/                  # CUTLASS headers
│   ├── examples/                 # Reference implementations
│   │   ├── 45_dual_gemm/        # Dual GEMM patterns
│   │   └── 72_blackwell_*/      # SM100 NVFP4 patterns
│   └── build-guide/                   # CMake build directory
├── src/                         # Custom kernel source
│   ├── dual_gemm_nvfp4.cuh      # CUTLASS kernel header
│   ├── dual_gemm_nvfp4.cu       # CUDA implementation
│   ├── pytorch_bindings.cpp     # PyTorch C++ bindings
│   ├── setup.py                 # Python build script
│   └── CMakeLists.txt           # CMake configuration
├── python/
│   └── submission.py            # Entry point
└── docs/
    ├── build.md                 # This file
    └── build-guide/
        ├── cmake-reference.md   # CMake configuration reference
        └── troubleshooting.md   # Common errors and solutions
```

---

## Next Steps

- See [CMake Reference](build-guide/cmake-reference.md) for advanced build configuration
- See [Troubleshooting](build-guide/troubleshooting.md) for common errors and solutions
