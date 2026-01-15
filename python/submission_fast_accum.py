# =============================================================================
# NVFP4 Dual-GEMM Submission - Fast Accum Version
# =============================================================================
# Applies use_fast_accum=True for potential throughput boost
# =============================================================================

import torch
from typing import Tuple

input_t = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]
output_t = torch.Tensor


def custom_kernel(data: input_t) -> output_t:
    """
    NVFP4 dual GEMM with use_fast_accum optimization.

    Key optimizations:
    - use_fast_accum=True for faster tensorcore accumulation
    - .t() instead of .T for view-based transpose
    - Write directly to pre-allocated c_out
    """
    a, b1, b2, _, _, _, sfa_perm, sfb1_perm, sfb2_perm, c_out = data

    m, n, l = c_out.shape

    if l == 1:
        # Scale factor conversion (GPU-only, no CPU transfers)
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)

        # Matrix slices - use .t() for view-based transpose
        a_mat = a[:, :, 0]
        b1_t = b1[:, :, 0].t()  # View, not copy
        b2_t = b2[:, :, 0].t()

        # Dual GEMM with fast accumulation
        r1 = torch._scaled_mm(
            a_mat, b1_t, scale_a, scale_b1,
            out_dtype=torch.float32,
            use_fast_accum=True  # Key optimization
        )
        r2 = torch._scaled_mm(
            a_mat, b2_t, scale_a, scale_b2,
            out_dtype=torch.float32,
            use_fast_accum=True
        )

        # Fused epilogue - silu(r1) * r2 -> fp16
        # silu(x) = x * sigmoid(x)
        c_out[:, :, 0] = (r1 * torch.sigmoid(r1) * r2).to(torch.float16)

        return c_out

    # L > 1 case
    for l_idx in range(l):
        scale_a = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)

        a_mat = a[:, :, l_idx]
        b1_t = b1[:, :, l_idx].t()
        b2_t = b2[:, :, l_idx].t()

        r1 = torch._scaled_mm(
            a_mat, b1_t, scale_a, scale_b1,
            out_dtype=torch.float32,
            use_fast_accum=True
        )
        r2 = torch._scaled_mm(
            a_mat, b2_t, scale_a, scale_b2,
            out_dtype=torch.float32,
            use_fast_accum=True
        )

        c_out[:, :, l_idx] = (r1 * torch.sigmoid(r1) * r2).to(torch.float16)

    return c_out
