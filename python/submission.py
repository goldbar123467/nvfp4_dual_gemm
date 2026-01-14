# =============================================================================
# NVFP4 Dual-GEMM Submission
# =============================================================================
# GPU MODE Leaderboard Submission
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
# =============================================================================

import torch
from typing import Tuple, Optional, Dict

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
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
    Optimized NVFP4 block-scaled dual GEMM with SiLU activation.

    Computes: C = silu(A @ B1) * (A @ B2)

    Optimizations:
    - GPU-based scale factor conversion using permuted format
    - Direct tensor operations without CPU transfers
    - Fused SiLU + multiply epilogue
    """
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c_out = data

    m, n, l = c_out.shape

    # Fast path for L=1 (common case)
    if l == 1:
        # Convert permuted -> blocked format on GPU
        # Input: [32, 4, rest_m, 4, rest_k, L] -> blocked format for cuBLAS
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)

        # Get matrix slices
        a_slice = a[:, :, 0]
        b1_t = b1[:, :, 0].T
        b2_t = b2[:, :, 0].T

        # Dual GEMM using NVFP4 tensor cores
        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        # Fused epilogue: silu(r1) * r2 -> FP16
        output = (torch.nn.functional.silu(r1) * r2).to(torch.float16)
        return output.unsqueeze(-1)

    # General case for L > 1
    output = torch.empty((m, n, l), dtype=torch.float16, device="cuda")

    for l_idx in range(l):
        scale_a = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)

        a_slice = a[:, :, l_idx]
        b1_t = b1[:, :, l_idx].T
        b2_t = b2[:, :, l_idx].T

        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        output[:, :, l_idx] = (torch.nn.functional.silu(r1) * r2).to(torch.float16)

    return output
