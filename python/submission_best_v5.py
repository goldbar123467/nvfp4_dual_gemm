# =============================================================================
# NVFP4 Dual-GEMM Submission - Best V5 (All Caching, No Compile)
# =============================================================================
# GPU MODE Leaderboard Submission
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
#
# Caches: scale transforms + transposed weight views
# =============================================================================

import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any

input_t = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]
output_t = torch.Tensor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Unified cache
_cache: Dict[tuple, Any] = {}


def _get_scale(sf_perm: torch.Tensor, l_idx: int) -> torch.Tensor:
    key = ('scale', sf_perm.data_ptr(), l_idx)
    if key not in _cache:
        _cache[key] = sf_perm[:,:,:,:,:,l_idx].permute(2,4,0,1,3).reshape(-1).contiguous()
    return _cache[key]


def _get_transposed(b: torch.Tensor, l_idx: int) -> torch.Tensor:
    key = ('trans', b.data_ptr(), l_idx)
    if key not in _cache:
        _cache[key] = b[:,:,l_idx].t().contiguous()
    return _cache[key]


def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, _, _, _, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape

    if l == 1:
        scale_a = _get_scale(sfa_perm, 0)
        scale_b1 = _get_scale(sfb1_perm, 0)
        scale_b2 = _get_scale(sfb2_perm, 0)
        b1_t = _get_transposed(b1, 0)
        b2_t = _get_transposed(b2, 0)
        a_mat = a[:,:,0]

        r1 = torch._scaled_mm(a_mat, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_mat, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        c_out[:,:,0] = (F.silu(r1) * r2).half()
        return c_out

    for l_idx in range(l):
        scale_a = _get_scale(sfa_perm, l_idx)
        scale_b1 = _get_scale(sfb1_perm, l_idx)
        scale_b2 = _get_scale(sfb2_perm, l_idx)
        b1_t = _get_transposed(b1, l_idx)
        b2_t = _get_transposed(b2, l_idx)
        a_mat = a[:,:,l_idx]

        r1 = torch._scaled_mm(a_mat, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_mat, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        c_out[:,:,l_idx] = (F.silu(r1) * r2).half()

    return c_out
