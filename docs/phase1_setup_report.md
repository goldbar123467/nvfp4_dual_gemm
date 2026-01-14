# Phase 1 Setup Report: CUTLASS for Blackwell SM100 Development

**Date**: 2026-01-14  
**Status**: Partially Complete (Source Code Ready, Build Environment Missing)

---

## 1. Executive Summary

CUTLASS 4.3.5 has been successfully cloned and contains all required Blackwell SM100 examples including the critical NVFP4 narrow precision GEMM examples (Example 72). However, the current development environment lacks the CUDA Toolkit and GPU hardware required to build and test the kernels.

---

## 2. CUTLASS Repository Status

### 2.1 Clone Location
```
/home/ubuntu/projects/nvfp4-dual-gemm/cutlass/
```

### 2.2 Version
- **CUTLASS Version**: 4.3.5 (January 2026)
- **Clone Type**: Shallow clone (--depth 1)

### 2.3 Key Files Located

#### Example 72 - Blackwell Narrow Precision GEMM
Location: `/home/ubuntu/projects/nvfp4-dual-gemm/cutlass/examples/72_blackwell_narrow_precision_gemm/`

| File | Description |
|------|-------------|
| `72a_blackwell_nvfp4_bf16_gemm.cu` | NVFP4 input, BF16 output GEMM |
| `72b_blackwell_nvfp4_nvfp4_gemm.cu` | NVFP4 input, NVFP4 output GEMM (most relevant for dual GEMM) |
| `72c_blackwell_mixed_mxfp8_bf16_gemm.cu` | Mixed precision MXFP8/BF16 GEMM |

#### Other Relevant Blackwell Examples
| Example | Path | Description |
|---------|------|-------------|
| 70 | `70_blackwell_gemm` | Basic Blackwell GEMM |
| 71 | `71_blackwell_gemm_with_collective_builder` | Collective builder pattern |
| 74 | `74_blackwell_gemm_streamk` | Stream-K scheduling |
| 79 | `79_blackwell_geforce_gemm` | GeForce-specific GEMM (incl. NVFP4) |
| 80 | `80_blackwell_geforce_sparse_gemm` | Sparse GEMM with NVFP4 |
| 84 | `84_blackwell_narrow_precision_sparse_gemm` | NVFP4 sparse GEMM |
| 86 | `86_blackwell_mixed_dtype_gemm` | Mixed datatype GEMM |
| 92 | `92_blackwell_moe_gemm` | MoE GEMM with FP4 variants |

---

## 3. System Environment Analysis

### 3.1 Operating System
```
Ubuntu 22.04.5 LTS (Jammy Jellyfish)
Kernel: Linux 5.15.0-164-generic
Platform: linux (x86_64)
```

### 3.2 Missing Dependencies

| Component | Status | Required Version | Notes |
|-----------|--------|------------------|-------|
| CUDA Toolkit | **NOT INSTALLED** | 12.8+ | Required for SM100 (Blackwell) |
| nvcc | **NOT FOUND** | CUDA 12.8+ | Not in PATH or standard locations |
| nvidia-smi | **NOT AVAILABLE** | - | No GPU driver detected |
| CMake | **NOT INSTALLED** | 3.18+ | Required for CUTLASS build |
| NVIDIA GPU | **NOT DETECTED** | B200/B300 | No GPU hardware visible |

### 3.3 Checked Locations for CUDA
- `/usr/local/cuda*` - Empty
- `/opt/cuda*` - Empty
- `/opt/nvidia*` - Empty
- System PATH - Not found
- dpkg packages - No CUDA packages installed

---

## 4. CUDA Requirements for Blackwell SM100

Per CUTLASS README:

| GPU | Compute Capability | Minimum CUDA Toolkit |
|-----|-------------------|---------------------|
| NVIDIA B200 Tensor Core GPU | 10.0 (SM100) | **CUDA 12.8** |
| NVIDIA B300 Tensor Core GPU | 10.3 (SM103) | CUDA 13.0 |

### 4.1 CMake Configuration (Not Yet Possible)

The intended CMake configuration for SM100:
```bash
mkdir -p build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=100 -DCUTLASS_ENABLE_EXAMPLES=ON
```

**Note**: Example 72 CMakeLists.txt shows it only builds when:
```cmake
if(CUTLASS_NVCC_ARCHS MATCHES "100a|100f|101a|101f|103a|103f")
```

This means we need to specify architecture variants:
- `100a` or `100f` for SM100 (B200)
- `103a` or `103f` for SM103 (B300)

---

## 5. Key Technical Insights from Example 72

From `72b_blackwell_nvfp4_nvfp4_gemm.cu`:

### 5.1 Features Used
1. **Blockscaled tcgen05.mma instructions** - Native FP4 tensor core operations
2. **Tensor Memory (TMEM)** - Per-SM memory for Blackwell
3. **Extended warp-specialized design** - Decoupled MMA and epilogue execution
4. **Cluster launch control** - SW-controlled dynamic scheduler

### 5.2 Kernel Characteristics
- Input: NVFP4 (e2m1) block-scaled format
- Output: NVFP4 (e2m1) with scale factors (can chain GEMMs)
- Optimized for Blackwell Tensor Cores

---

## 6. Blockers and Required Actions

### 6.1 Critical Blockers

| Priority | Blocker | Resolution |
|----------|---------|------------|
| **P0** | No CUDA Toolkit | Install CUDA 12.8+ |
| **P0** | No CMake | Install CMake 3.18+ |
| **P0** | No GPU Hardware | Provision B200/B300 GPU instance |

### 6.2 Required Installation Steps

1. **Install CUDA 12.8 Toolkit**:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_560.35.03_linux.run
   sudo sh cuda_12.8.0_560.35.03_linux.run
   export PATH=/usr/local/cuda-12.8/bin:$PATH
   export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
   ```

2. **Install CMake**:
   ```bash
   sudo apt-get update
   sudo apt-get install cmake
   ```

3. **Verify GPU Access**:
   ```bash
   nvidia-smi
   nvcc --version
   ```

---

## 7. Next Steps (Once Environment Ready)

1. Configure CMake for SM100:
   ```bash
   cd /home/ubuntu/projects/nvfp4-dual-gemm/cutlass
   mkdir -p build && cd build
   cmake .. -DCUTLASS_NVCC_ARCHS=100a -DCUTLASS_ENABLE_EXAMPLES=ON
   make -j$(nproc) 72b_blackwell_nvfp4_nvfp4_gemm
   ```

2. Run Example 72b:
   ```bash
   ./examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm --m=2048 --n=2048 --k=2048
   ```

3. Study the implementation for dual GEMM fusion approach

---

## 8. Files and Directories Created

| Path | Type | Description |
|------|------|-------------|
| `/home/ubuntu/projects/nvfp4-dual-gemm/cutlass/` | Directory | CUTLASS 4.3.5 repository |
| `/home/ubuntu/projects/nvfp4-dual-gemm/docs/phase1_setup_report.md` | File | This report |

---

## 9. Conclusion

The CUTLASS source code is ready and contains all the Blackwell SM100 NVFP4 examples needed for the dual GEMM project. The primary blockers are:

1. **Missing CUDA Toolkit 12.8+** - Required for SM100/Blackwell compilation
2. **Missing CMake** - Required for CUTLASS build system
3. **No GPU Hardware Detected** - Need B200 or compatible Blackwell GPU

Once a proper Blackwell-capable environment is provisioned with CUDA 12.8+, the project can proceed to building and studying Example 72b as the foundation for the dual GEMM kernel.

---

*Report generated: 2026-01-14*
