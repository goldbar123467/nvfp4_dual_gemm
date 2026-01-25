# =============================================================================
# NVFP4 Dual-GEMM Submission - torch.compile on full kernel
# =============================================================================
# GPU MODE Leaderboard Submission
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
#
# Uses torch.compile on the entire kernel for maximum fusion
# =============================================================================

import torch
from typing import Tuple

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


@torch.compile(mode="reduce-overhead")
def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, _, _, _, sfa_perm, sfb1_perm, sfb2_perm, c_out = data

    m, n, l = c_out.shape

    if l == 1:
        scale_a  = sfa_perm[:,:,:,:,:,0].permute(2,4,0,1,3).reshape(-1)
        scale_b1 = sfb1_perm[:,:,:,:,:,0].permute(2,4,0,1,3).reshape(-1)
        scale_b2 = sfb2_perm[:,:,:,:,:,0].permute(2,4,0,1,3).reshape(-1)

        a_mat = a[:,:,0]
        b1_t  = b1[:,:,0].t()
        b2_t  = b2[:,:,0].t()

        r1 = torch._scaled_mm(a_mat, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_mat, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        # fused epilogue — the compiler should do a good job here
        c_out[:,:,0].copy_(torch.nn.functional.silu(r1) * r2.to(r1.dtype))

        return c_out

    # l > 1 case
    for l_idx in range(l):
        scale_a  = sfa_perm[:,:,:,:,:,l_idx].permute(2,4,0,1,3).reshape(-1)
        scale_b1 = sfb1_perm[:,:,:,:,:,l_idx].permute(2,4,0,1,3).reshape(-1)
        scale_b2 = sfb2_perm[:,:,:,:,:,l_idx].permute(2,4,0,1,3).reshape(-1)

        a_mat = a[:,:,l_idx]
        b1_t  = b1[:,:,l_idx].t()
        b2_t  = b2[:,:,l_idx].t()

        r1 = torch._scaled_mm(a_mat, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_mat, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        c_out[:,:,l_idx].copy_(torch.nn.functional.silu(r1) * r2.to(r1.dtype))

    return c_out
