# =============================================================================
# NVFP4 Dual-GEMM Submission - Best V5 (Cache views only)
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

_cache: Dict[tuple, Any] = {}


def _get_scale(sf_perm: torch.Tensor, l_idx: int) -> torch.Tensor:
    key = ('s', sf_perm.data_ptr(), l_idx)
    if key not in _cache:
        # Don't call contiguous - just reshape
        _cache[key] = sf_perm[:,:,:,:,:,l_idx].permute(2,4,0,1,3).reshape(-1)
    return _cache[key]


def _get_transposed(b: torch.Tensor, l_idx: int) -> torch.Tensor:
    key = ('t', b.data_ptr(), l_idx)
    if key not in _cache:
        # Just the view, no contiguous copy
        _cache[key] = b[:,:,l_idx].t()
    return _cache[key]


def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, _, _, _, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape

    if l == 1:
        r1 = torch._scaled_mm(
            a[:,:,0],
            _get_transposed(b1, 0),
            _get_scale(sfa_perm, 0),
            _get_scale(sfb1_perm, 0),
            out_dtype=torch.float32
        )
        r2 = torch._scaled_mm(
            a[:,:,0],
            _get_transposed(b2, 0),
            _get_scale(sfa_perm, 0),
            _get_scale(sfb2_perm, 0),
            out_dtype=torch.float32
        )
        c_out[:,:,0] = (F.silu(r1) * r2).half()
        return c_out

    for l_idx in range(l):
        r1 = torch._scaled_mm(
            a[:,:,l_idx],
            _get_transposed(b1, l_idx),
            _get_scale(sfa_perm, l_idx),
            _get_scale(sfb1_perm, l_idx),
            out_dtype=torch.float32
        )
        r2 = torch._scaled_mm(
            a[:,:,l_idx],
            _get_transposed(b2, l_idx),
            _get_scale(sfa_perm, l_idx),
            _get_scale(sfb2_perm, l_idx),
            out_dtype=torch.float32
        )
        c_out[:,:,l_idx] = (F.silu(r1) * r2).half()

    return c_out
