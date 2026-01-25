# =============================================================================
# NVFP4 Dual-GEMM Submission - FP16 Output Direct
# =============================================================================
# GPU MODE Leaderboard Submission
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
#
# Uses FP16 output from scaled_mm directly, minimal operations
# =============================================================================

import torch
import torch.nn.functional as F
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


def custom_kernel(data: input_t) -> output_t:
    """
    NVFP4 dual GEMM - FP16 output, minimal ops.
    """
    a, b1, b2, _, _, _, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape

    if l == 1:
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)

        a0 = a[:, :, 0]
        b1t = b1[:, :, 0].t()
        b2t = b2[:, :, 0].t()

        # Try FP16 output directly
        r1 = torch._scaled_mm(a0, b1t, scale_a, scale_b1, out_dtype=torch.float16)
        r2 = torch._scaled_mm(a0, b2t, scale_a, scale_b2, out_dtype=torch.float16)

        # silu in fp16
        c_out[:, :, 0] = F.silu(r1) * r2
        return c_out

    for i in range(l):
        scale_a = sfa_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1)

        a_i = a[:, :, i]
        b1t = b1[:, :, i].t()
        b2t = b2[:, :, i].t()

        r1 = torch._scaled_mm(a_i, b1t, scale_a, scale_b1, out_dtype=torch.float16)
        r2 = torch._scaled_mm(a_i, b2t, scale_a, scale_b2, out_dtype=torch.float16)

        c_out[:, :, i] = F.silu(r1) * r2

    return c_out
