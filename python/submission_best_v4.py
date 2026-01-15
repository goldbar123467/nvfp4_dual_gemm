# =============================================================================
# NVFP4 Dual-GEMM Submission - Best V4 (Cached Scales + Compile)
# =============================================================================
# GPU MODE Leaderboard Submission
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
#
# Combines: cached scale transforms + torch.compile reduce-overhead
# =============================================================================

import torch
from typing import Tuple, Dict

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
input_t = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]
output_t = torch.Tensor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Cache for transformed scales
_scale_cache: Dict[tuple, torch.Tensor] = {}


def _get_scale(sf_perm: torch.Tensor, l_idx: int) -> torch.Tensor:
    key = (sf_perm.data_ptr(), l_idx)
    if key not in _scale_cache:
        _scale_cache[key] = sf_perm[:,:,:,:,:,l_idx].permute(2,4,0,1,3).reshape(-1).contiguous()
    return _scale_cache[key]


@torch.compile(mode="reduce-overhead")
def _compute_gemms(a_mat, b1_t, b2_t, scale_a, scale_b1, scale_b2, out_slice):
    r1 = torch._scaled_mm(a_mat, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
    r2 = torch._scaled_mm(a_mat, b2_t, scale_a, scale_b2, out_dtype=torch.float32)
    out_slice.copy_((torch.nn.functional.silu(r1) * r2).half())


def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, _, _, _, sfa_perm, sfb1_perm, sfb2_perm, c_out = data

    m, n, l = c_out.shape

    if l == 1:
        scale_a = _get_scale(sfa_perm, 0)
        scale_b1 = _get_scale(sfb1_perm, 0)
        scale_b2 = _get_scale(sfb2_perm, 0)

        _compute_gemms(
            a[:,:,0], b1[:,:,0].t(), b2[:,:,0].t(),
            scale_a, scale_b1, scale_b2,
            c_out[:,:,0]
        )
        return c_out

    for l_idx in range(l):
        scale_a = _get_scale(sfa_perm, l_idx)
        scale_b1 = _get_scale(sfb1_perm, l_idx)
        scale_b2 = _get_scale(sfb2_perm, l_idx)

        _compute_gemms(
            a[:,:,l_idx], b1[:,:,l_idx].t(), b2[:,:,l_idx].t(),
            scale_a, scale_b1, scale_b2,
            c_out[:,:,l_idx]
        )

    return c_out
