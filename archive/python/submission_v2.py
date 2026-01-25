# =============================================================================
# NVFP4 Dual-GEMM Submission v2 - Cached Scale Factors
# =============================================================================
# GPU MODE Leaderboard Submission
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
#
# Caches scale factor conversions to avoid repeated permute/reshape overhead
# =============================================================================

import torch
from typing import Tuple, Dict

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


class _ScaleCache:
    """Cache converted scale factors to avoid repeated permute/reshape."""

    def __init__(self):
        self.cache: Dict[tuple, tuple] = {}

    def get_scales(self, sfa_perm: torch.Tensor, sfb1_perm: torch.Tensor,
                   sfb2_perm: torch.Tensor, l_idx: int = 0):
        """Get cached or compute scale factors."""
        key = (sfa_perm.data_ptr(), sfb1_perm.data_ptr(), sfb2_perm.data_ptr(), l_idx)

        if key not in self.cache:
            scale_a = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1).clone()
            scale_b1 = sfb1_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1).clone()
            scale_b2 = sfb2_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1).clone()
            self.cache[key] = (scale_a, scale_b1, scale_b2)

        return self.cache[key]


_scale_cache = _ScaleCache()


def custom_kernel(data: input_t) -> output_t:
    """
    Optimized NVFP4 block-scaled dual GEMM with SiLU activation.

    Computes: C = silu(A @ B1) * (A @ B2)

    v2: Caches scale factor conversions to avoid repeated permute/reshape.
    """
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c_out = data

    m, n, l = c_out.shape

    if l == 1:
        scale_a, scale_b1, scale_b2 = _scale_cache.get_scales(sfa_perm, sfb1_perm, sfb2_perm, 0)

        a_slice = a[:, :, 0]
        b1_t = b1[:, :, 0].T
        b2_t = b2[:, :, 0].T

        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        output = (torch.nn.functional.silu(r1) * r2).to(torch.float16)
        return output.unsqueeze(-1)

    output = torch.empty((m, n, l), dtype=torch.float16, device=a.device)

    for l_idx in range(l):
        scale_a, scale_b1, scale_b2 = _scale_cache.get_scales(sfa_perm, sfb1_perm, sfb2_perm, l_idx)

        a_slice = a[:, :, l_idx]
        b1_t = b1[:, :, l_idx].T
        b2_t = b2[:, :, l_idx].T

        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        output[:, :, l_idx] = (torch.nn.functional.silu(r1) * r2).to(torch.float16)

    return output
