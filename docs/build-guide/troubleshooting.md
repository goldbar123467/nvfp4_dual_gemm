# Troubleshooting Guide

This document covers common build errors and their solutions for the NVFP4 dual-GEMM kernel.

---

## Quick Diagnostic

Run this script to diagnose common issues:

```bash
#!/bin/bash
echo "=== Environment Check ==="
echo "CUDA version:"
nvcc --version 2>&1 | head -4

echo -e "\nGPU compute capability:"
nvidia-smi --query-gpu=compute_cap --format=csv 2>&1

echo -e "\nPyTorch CUDA:"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" 2>&1

echo -e "\nCUTLASS path:"
ls -la /workspace/nvfp4_dual_gemm/cutlass/include/cutlass/cutlass.h 2>&1

echo -e "\nGCC version:"
gcc --version 2>&1 | head -1
```

---

## Common Errors and Solutions

### 1. CUDA Version Mismatch

**Error:**
```
error: This example requires CUDA 12.8 or newer.
```

**Cause:** CUDA Toolkit version is older than 12.8.

**Solution:**
```bash
# Check current version
nvcc --version

# If outdated, update CUDA toolkit
# Download from: https://developer.nvidia.com/cuda-downloads
# Select Linux > x86_64 > Ubuntu > 22.04 > runfile

# Or use module system if available
module load cuda/12.8
```

---

### 2. Compute Capability Error

**Error:**
```
error: requires a GPU with compute capability 100a|f, 101a|f, or 103a|f
```

**Cause:** Running on a GPU that's not SM100 Blackwell architecture.

**Solution:**
```bash
# Check GPU
nvidia-smi -L

# Verify compute capability (must show 10.x)
nvidia-smi --query-gpu=compute_cap --format=csv

# If wrong GPU, set CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0  # Adjust to B200 device ID
```

---

### 3. CUTLASS_ARCH_MMA_SM100_SUPPORTED Not Defined

**Error:**
```
error: use of undefined macro 'CUTLASS_ARCH_MMA_SM100_SUPPORTED'
```
or
```
#error This file requires Blackwell (SM100) or later architectures
```

**Cause:** Missing compiler definition for SM100 support.

**Solution:**

For setup.py:
```python
extra_compile_args={
    'nvcc': [
        '-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1',
        # ... other flags
    ]
}
```

For CMake:
```cmake
target_compile_definitions(target PRIVATE
    CUTLASS_ARCH_MMA_SM100_SUPPORTED=1
)
```

For manual compilation:
```bash
nvcc -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1 ...
```

---

### 4. C++17 Standard Required

**Error:**
```
error: 'constexpr' function return is non-constant
```
or
```
error: static assertion failed: CUTLASS requires C++17 or later
```

**Cause:** Compiler not using C++17 standard.

**Solution:**

For NVCC:
```bash
nvcc -std=c++17 --expt-relaxed-constexpr ...
```

For setup.py:
```python
extra_compile_args={
    'cxx': ['-std=c++17'],
    'nvcc': ['-std=c++17', '--expt-relaxed-constexpr']
}
```

For CMake:
```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
```

---

### 5. Scale Factor Layout Mismatch

**Error:**
```
error: no member named 'from_shape_params' in 'LayoutSFA'
```
or
```
error: no matching function for call to 'from_shape_params'
```

**Cause:** Using incorrect API for scale factor layout construction.

**Solution:**

Use `tile_atom_to_shape_SFA` instead:
```cpp
// WRONG
auto layout_SFA = LayoutSFA::from_shape_params(M, K);

// CORRECT
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
    cute::make_shape(M, N, K, 1));
```

---

### 6. CUTLASS Include Path Not Found

**Error:**
```
fatal error: cutlass/cutlass.h: No such file or directory
```
or
```
fatal error: cute/tensor.hpp: No such file or directory
```

**Cause:** CUTLASS include path not set correctly.

**Solution:**

Verify CUTLASS location:
```bash
ls -la /workspace/nvfp4_dual_gemm/cutlass/include/cutlass/cutlass.h
```

For setup.py:
```python
import os
cutlass_include = os.path.join(os.path.dirname(__file__), '..', 'cutlass', 'include')
assert os.path.exists(cutlass_include), f"CUTLASS not found at {cutlass_include}"

CUDAExtension(
    ...,
    include_dirs=[cutlass_include],
)
```

For CMake:
```cmake
set(CUTLASS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include")
if(NOT EXISTS "${CUTLASS_INCLUDE_DIR}/cutlass/cutlass.h")
    message(FATAL_ERROR "CUTLASS not found at ${CUTLASS_INCLUDE_DIR}")
endif()
```

---

### 7. PyTorch/CUDA Version Incompatibility

**Error:**
```
RuntimeError: Detected that PyTorch and torchvision were compiled with different CUDA versions
```
or
```
undefined symbol: __cudaRegisterFatBinary
```

**Cause:** PyTorch compiled with different CUDA version than system CUDA.

**Solution:**

Check versions:
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"PyTorch cuDNN: {torch.backends.cudnn.version()}")
```

```bash
nvcc --version
```

If mismatch, reinstall PyTorch with matching CUDA:
```bash
# For CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

---

### 8. Linker Errors - Undefined Symbols

**Error:**
```
undefined reference to `cutlass::Status::to_string()'
```
or
```
undefined symbol: _ZN7cutlass...
```

**Cause:** Missing library linkage or ABI incompatibility.

**Solution:**

For CMake, ensure proper linking:
```cmake
target_link_libraries(target PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
)
```

Check ABI compatibility:
```bash
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
```

Add to compile flags if needed:
```cmake
target_compile_definitions(target PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0  # or 1, match PyTorch
)
```

---

### 9. Out of Memory During Compilation

**Error:**
```
nvcc fatal: Virtual memory exhausted
```
or
```
c++: fatal error: Killed signal terminated program cc1plus
```

**Cause:** CUTLASS template instantiation consuming too much memory.

**Solution:**

Reduce parallel jobs:
```bash
make -j4  # Instead of -j$(nproc)
```

Increase swap space:
```bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

Split compilation units:
```cmake
set_target_properties(target PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
```

---

### 10. Architecture Mismatch at Runtime

**Error:**
```
CUDA error: no kernel image is available for execution on the device
```

**Cause:** Kernel compiled for different GPU architecture.

**Solution:**

Verify compilation architecture:
```bash
cuobjdump -arch sm_100a your_kernel.so
```

Recompile with correct architecture:
```bash
nvcc -arch=sm_100a ...
```

For fat binaries (multiple architectures):
```bash
nvcc -gencode=arch=compute_100,code=sm_100 ...
```

---

## Advanced Troubleshooting

### Enable Verbose Compilation

```bash
# setup.py
python setup.py build_ext --inplace --verbose

# CMake
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
make VERBOSE=1

# NVCC
nvcc --verbose ...
```

### Debug CUTLASS Template Instantiation

```bash
# Show full template expansion
nvcc --expt-relaxed-constexpr -Xcompiler=-ftemplate-backtrace-limit=0 ...
```

### Check PTX Assembly

```bash
# Generate PTX
nvcc -ptx -arch=sm_100a kernel.cu -o kernel.ptx

# Inspect generated code
cat kernel.ptx | grep -A 10 "mma.sync"
```

### Profile with NSight

```bash
# Build with profiling support
nvcc -lineinfo --generate-line-info ...

# Run profiler
ncu --set full ./your_binary
```

---

## Getting Help

If you encounter issues not covered here:

1. Check CUTLASS GitHub issues: https://github.com/NVIDIA/cutlass/issues
2. Review Example 72 source code for patterns
3. Verify against the reference implementation in `cutlass/examples/72_blackwell_narrow_precision_gemm/`

### Collecting Debug Information

When reporting issues, include:
```bash
# System info
uname -a
nvcc --version
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda)"

# Full error output
python setup.py build_ext --inplace 2>&1 | tee build.log
```
