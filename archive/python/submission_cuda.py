# =============================================================================
# NVFP4 Dual-GEMM Submission - CUDA C++ Extension
# =============================================================================
# Uses custom C++ extension with fused SiLU*mul kernel
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
# =============================================================================

import sys
import os

# Add src directory for the extension
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, src_path)

import torch
from typing import Tuple

try:
    import nvfp4_dual_gemm_cuda
    HAS_CUDA_EXT = True
except ImportError:
    HAS_CUDA_EXT = False
    print("WARNING: nvfp4_dual_gemm_cuda not found. Build with: cd src && pip install -e .")

# Type aliases
input_t = Tuple[
    torch.Tensor,  # a: [M, K//2, L] FP4 packed
    torch.Tensor,  # b1: [N, K//2, L] FP4 packed
    torch.Tensor,  # b2: [N, K//2, L] FP4 packed
    torch.Tensor,  # sfa: [M, K//16, L] FP8 scale factors
    torch.Tensor,  # sfb1: [N, K//16, L] FP8 scale factors
    torch.Tensor,  # sfb2: [N, K//16, L] FP8 scale factors
    torch.Tensor,  # sfa_permuted: [32, 4, rest_m, 4, rest_k, L]
    torch.Tensor,  # sfb1_permuted
    torch.Tensor,  # sfb2_permuted
    torch.Tensor,  # c: [M, N, L] FP16 output
]
output_t = torch.Tensor


def custom_kernel(data: input_t) -> output_t:
    """
    NVFP4 block-scaled dual GEMM with SiLU activation using C++ extension.

    Computes: C = silu(A @ B1) * (A @ B2)

    Optimizations:
    - C++ extension eliminates Python dispatch overhead
    - Custom fused SiLU*mul CUDA kernel
    - Uses pre-permuted scale factors (GPU-only)
    """
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape

    if l == 1:
        # Convert permuted -> blocked format on GPU
        # Input: [32, 4, rest_m, 4, rest_k, L] -> blocked format for cuBLAS
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)

        # Call C++ extension
        result = nvfp4_dual_gemm_cuda.dual_gemm_silu(
            a[:, :, 0], b1[:, :, 0], b2[:, :, 0],
            scale_a, scale_b1, scale_b2
        )
        return result.unsqueeze(-1)

    # General case for L > 1
    output = torch.empty((m, n, l), dtype=torch.float16, device="cuda")

    for l_idx in range(l):
        scale_a = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)

        result = nvfp4_dual_gemm_cuda.dual_gemm_silu(
            a[:, :, l_idx], b1[:, :, l_idx], b2[:, :, l_idx],
            scale_a, scale_b1, scale_b2
        )
        output[:, :, l_idx] = result

    return output
