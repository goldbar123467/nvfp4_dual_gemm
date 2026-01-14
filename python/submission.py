# =============================================================================
# NVFP4 Dual-GEMM Submission - torch.compile optimized
# =============================================================================
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
# =============================================================================

import torch
from typing import Tuple

# Type aliases
input_t = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]
output_t = torch.Tensor


def _dual_gemm_silu_inner(a_slice, b1_t, b2_t, scale_a, scale_b1, scale_b2):
    """Inner kernel - compiled with torch.compile."""
    # Dual GEMM using NVFP4 tensor cores
    r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
    r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

    # Fused epilogue: silu(r1) * r2 -> FP16
    return (torch.nn.functional.silu(r1) * r2).to(torch.float16)


# Try to compile the inner kernel
try:
    _compiled_kernel = torch.compile(_dual_gemm_silu_inner, mode="max-autotune")
except Exception:
    _compiled_kernel = _dual_gemm_silu_inner


def custom_kernel(data: input_t) -> output_t:
    """
    Optimized NVFP4 block-scaled dual GEMM with SiLU activation.

    Computes: C = silu(A @ B1) * (A @ B2)
    """
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape

    if l == 1:
        # Convert permuted -> blocked scale factors
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)

        a_slice = a[:, :, 0]
        b1_t = b1[:, :, 0].T
        b2_t = b2[:, :, 0].T

        output = _compiled_kernel(a_slice, b1_t, b2_t, scale_a, scale_b1, scale_b2)
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

        output[:, :, l_idx] = _compiled_kernel(a_slice, b1_t, b2_t, scale_a, scale_b1, scale_b2)

    return output
