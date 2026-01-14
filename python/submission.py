# =============================================================================
# NVFP4 Dual-GEMM Submission - Maximum PyTorch Optimization
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

# Enable all available optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


def custom_kernel(data: input_t) -> output_t:
    """
    Maximum optimized NVFP4 block-scaled dual GEMM with SiLU activation.

    Computes: C = silu(A @ B1) * (A @ B2)
    """
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape

    if l == 1:
        # Pre-compute contiguous scale factors (avoid lazy evaluation)
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1).contiguous()
        scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1).contiguous()
        scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1).contiguous()

        # Get contiguous matrix slices - avoid strided access
        a_slice = a[:, :, 0].contiguous()
        b1_t = b1[:, :, 0].T.contiguous()
        b2_t = b2[:, :, 0].T.contiguous()

        # Dual GEMM using NVFP4 tensor cores
        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        # Fused epilogue: silu(r1) * r2 -> FP16
        # silu(x) = x * sigmoid(x)
        output = torch.mul(torch.nn.functional.silu(r1), r2).half()
        return output.unsqueeze(-1)

    # General case for L > 1
    output = torch.empty((m, n, l), dtype=torch.float16, device=a.device)

    for l_idx in range(l):
        scale_a = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1).contiguous()
        scale_b1 = sfb1_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1).contiguous()
        scale_b2 = sfb2_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1).contiguous()

        a_slice = a[:, :, l_idx].contiguous()
        b1_t = b1[:, :, l_idx].T.contiguous()
        b2_t = b2[:, :, l_idx].T.contiguous()

        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        output[:, :, l_idx] = torch.mul(torch.nn.functional.silu(r1), r2).half()

    return output
