# CMake Configuration Reference

This document provides detailed CMake configuration options for building the NVFP4 dual-GEMM kernel.

---

## CUTLASS Build Variables

### Required Variables

```cmake
# Target SM100 Blackwell architecture
set(CUTLASS_NVCC_ARCHS "100a")

# Enable examples (required for reference builds)
set(CUTLASS_ENABLE_EXAMPLES ON)

# Disable tests for faster builds
set(CUTLASS_ENABLE_TESTS OFF)
```

### Optional Variables

```cmake
# Build type (Release recommended for performance)
set(CMAKE_BUILD_TYPE Release)

# Custom CUDA path (if not in system path)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.8/bin/nvcc)

# Enable verbose output
set(CMAKE_VERBOSE_MAKEFILE ON)
```

---

## PyTorch Extension CMakeLists.txt

### Full Configuration

```cmake
cmake_minimum_required(VERSION 3.18)
project(dual_gemm_nvfp4 LANGUAGES CXX CUDA)

# =============================================================================
# Package Discovery
# =============================================================================
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# =============================================================================
# Configuration
# =============================================================================
# CUTLASS include path
set(CUTLASS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cutlass/include")

# CUDA architecture for SM100 Blackwell
set(CMAKE_CUDA_ARCHITECTURES 100)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =============================================================================
# Extension Library
# =============================================================================
add_library(dual_gemm_nvfp4_ext SHARED
    dual_gemm_nvfp4.cu
    pytorch_bindings.cpp
)

# Include directories
target_include_directories(dual_gemm_nvfp4_ext PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUTLASS_INCLUDE_DIR}/../tools/util/include
    ${TORCH_INCLUDE_DIRS}
    ${Python_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Compile options
target_compile_options(dual_gemm_nvfp4_ext PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -std=c++17
        -fPIC
    >
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -arch=sm_100a
        --use_fast_math
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -lineinfo
        -Xcompiler=-fPIC
        -DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1
    >
)

# Compile definitions
target_compile_definitions(dual_gemm_nvfp4_ext PRIVATE
    CUTLASS_ARCH_MMA_SM100_SUPPORTED=1
)

# Link libraries
target_link_libraries(dual_gemm_nvfp4_ext PRIVATE
    ${TORCH_LIBRARIES}
    Python::Python
    CUDA::cudart
)

# Output properties
set_target_properties(dual_gemm_nvfp4_ext PROPERTIES
    PREFIX ""
    SUFFIX ".so"
    OUTPUT_NAME "dual_gemm_nvfp4_cuda"
    CXX_STANDARD 17
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
```

---

## NVCC Compiler Flags Reference

### Required Flags

| Flag | Purpose |
|------|---------|
| `-arch=sm_100a` | Target SM100 with all features |
| `-std=c++17` | C++17 standard (required for CUTLASS 3.x) |
| `--expt-relaxed-constexpr` | Extended constexpr support |
| `-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1` | Enable SM100 MMA support |

### Performance Flags

| Flag | Purpose |
|------|---------|
| `-O3` | Maximum optimization |
| `--use_fast_math` | Enable fast math operations |
| `-Xptxas -dlcm=ca` | Cache all memory loads |
| `--maxrregcount=128` | Limit register usage |

### Debug/Profiling Flags

| Flag | Purpose |
|------|---------|
| `-lineinfo` | Include line info for profiling |
| `-G` | Debug mode (disables optimization) |
| `-Xcompiler -rdynamic` | Dynamic symbols for profiling |

### Additional Flags

| Flag | Purpose |
|------|---------|
| `--expt-extended-lambda` | Extended lambda support |
| `-Xcompiler -fPIC` | Position independent code |
| `--generate-line-info` | Line info for nsight-compute |

---

## Build Configurations

### Debug Build

```cmake
set(CMAKE_BUILD_TYPE Debug)

target_compile_options(dual_gemm_nvfp4_ext PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -G                          # Debug mode
        -lineinfo
        -DDEBUG
    >
)
```

### Release Build (Recommended)

```cmake
set(CMAKE_BUILD_TYPE Release)

target_compile_options(dual_gemm_nvfp4_ext PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        -DNDEBUG
    >
)
```

### Profile Build

```cmake
set(CMAKE_BUILD_TYPE RelWithDebInfo)

target_compile_options(dual_gemm_nvfp4_ext PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -lineinfo
        --generate-line-info
        -Xcompiler -rdynamic
    >
)
```

---

## CUTLASS Example Targets

### Available Targets

```bash
# List all available targets
cd cutlass/build && make help | grep blackwell

# Key targets for this project:
make 72a_blackwell_nvfp4_bf16_gemm      # NVFP4 -> BF16 GEMM
make 72b_blackwell_nvfp4_nvfp4_gemm     # NVFP4 -> NVFP4 GEMM (reference)
make 72c_blackwell_mixed_mxfp8_bf16_gemm # MXFP8 -> BF16 GEMM
make 45_dual_gemm                        # Dual GEMM pattern
```

### Build All Examples

```bash
cd cutlass/build
make -j$(nproc) 72_blackwell_narrow_precision_gemm
```

---

## Integration with setup.py

The CMake configuration can be used alongside `setup.py` for complex builds:

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import subprocess

# Run CMake pre-build if needed
def run_cmake():
    build_dir = 'build_cmake'
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(['cmake', '..'], cwd=build_dir, check=True)
    subprocess.run(['make', '-j4'], cwd=build_dir, check=True)

cutlass_include = os.path.join(os.path.dirname(__file__), '..', 'cutlass', 'include')

setup(
    name='dual_gemm_nvfp4_cuda',
    ext_modules=[
        CUDAExtension(
            'dual_gemm_nvfp4_cuda',
            ['dual_gemm_nvfp4.cu', 'pytorch_bindings.cpp'],
            include_dirs=[cutlass_include],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_100a',
                    '-std=c++17',
                    '--use_fast_math',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-lineinfo',
                    '-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1',
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

---

## Environment Variables

### CMake Configuration

```bash
# PyTorch CMake prefix path
export CMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

# CUDA root
export CUDA_HOME=/usr/local/cuda-12.8

# CUTLASS root
export CUTLASS_PATH=/workspace/nvfp4_dual_gemm/cutlass
```

### Build Commands

```bash
# Full CMake build
cd /workspace/nvfp4_dual_gemm/src
mkdir -p build && cd build
cmake .. \
    -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=100
make -j$(nproc)
```
