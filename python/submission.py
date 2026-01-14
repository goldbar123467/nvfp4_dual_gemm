# =============================================================================
# NVFP4 Dual-GEMM Submission - Triton Epilogue
# =============================================================================
# Uses torch._scaled_mm for GEMMs + Triton fused epilogue
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
# =============================================================================

import torch
from typing import Tuple

# Try to import triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Type aliases
input_t = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]
output_t = torch.Tensor


if HAS_TRITON:
    @triton.jit
    def _silu_mul_kernel(
        output_ptr,
        input1_ptr,
        input2_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused SiLU(input1) * input2 kernel."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load inputs (FP32)
        x1 = tl.load(input1_ptr + offsets, mask=mask)
        x2 = tl.load(input2_ptr + offsets, mask=mask)

        # SiLU(x1) = x1 * sigmoid(x1)
        sigmoid_x1 = tl.sigmoid(x1)
        silu_x1 = x1 * sigmoid_x1

        # Output = SiLU(x1) * x2, convert to FP16
        output = silu_x1 * x2

        tl.store(output_ptr + offsets, output.to(tl.float16), mask=mask)

    def silu_mul_triton(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Apply fused SiLU * mul using Triton."""
        output = torch.empty(input1.shape, dtype=torch.float16, device=input1.device)
        n_elements = input1.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _silu_mul_kernel[grid](
            output, input1, input2, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return output


def custom_kernel(data: input_t) -> output_t:
    """
    NVFP4 block-scaled dual GEMM with SiLU activation.

    Computes: C = silu(A @ B1) * (A @ B2)

    Uses torch._scaled_mm for GEMMs + Triton fused epilogue.
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

        # Dual GEMM using NVFP4 tensor cores
        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        # Fused epilogue: silu(r1) * r2 -> FP16
        if HAS_TRITON:
            output = silu_mul_triton(r1, r2)
        else:
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

        if HAS_TRITON:
            output[:, :, l_idx] = silu_mul_triton(r1, r2)
        else:
            output[:, :, l_idx] = (torch.nn.functional.silu(r1) * r2).to(torch.float16)

    return output
