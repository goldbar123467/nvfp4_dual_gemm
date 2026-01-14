# =============================================================================
# NVFP4 Dual-GEMM Kernel Implementation
# =============================================================================
# Custom kernel: C = silu(A @ B1) * (A @ B2)
# =============================================================================

import torch
from typing import Tuple
from task import input_t, output_t, to_blocked

# Import from single source of truth
from constants import (
    OUTPUT_DTYPE,
    ACCUMULATOR_DTYPE,
)


def custom_kernel(data: input_t) -> output_t:
    """
    Optimized implementation of NVFP4 block-scaled dual GEMM with SiLU.

    Computes: C = silu(A @ B1) * (A @ B2)

    This implementation uses torch._scaled_mm for NVFP4 tensor core ops.
    """
    a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu, sfa_perm, sfb1_perm, sfb2_perm, c_out = data

    # Get dimensions from output shape (M, N, L)
    m, n, l = c_out.shape

    # Pre-allocate output tensors for both GEMMs
    result1 = torch.empty((l, m, n), dtype=ACCUMULATOR_DTYPE, device="cuda")
    result2 = torch.empty((l, m, n), dtype=ACCUMULATOR_DTYPE, device="cuda")

    # Process each batch
    for l_idx in range(l):
        # Convert scale factors to blocked format (cuBLAS layout)
        scale_a = to_blocked(sfa_cpu[:, :, l_idx]).cuda()
        scale_b1 = to_blocked(sfb1_cpu[:, :, l_idx]).cuda()
        scale_b2 = to_blocked(sfb2_cpu[:, :, l_idx]).cuda()

        # Extract batch slices
        a_slice = a[:, :, l_idx]           # [M, K//2]
        b1_slice = b1[:, :, l_idx]         # [N, K//2]
        b2_slice = b2[:, :, l_idx]         # [N, K//2]

        # GEMM1: A @ B1^T -> [M, N]
        result1[l_idx] = torch._scaled_mm(
            a_slice,
            b1_slice.transpose(0, 1),
            scale_a,
            scale_b1,
            bias=None,
            out_dtype=ACCUMULATOR_DTYPE,
        )

        # GEMM2: A @ B2^T -> [M, N]
        result2[l_idx] = torch._scaled_mm(
            a_slice,
            b2_slice.transpose(0, 1),
            scale_a,
            scale_b2,
            bias=None,
            out_dtype=ACCUMULATOR_DTYPE,
        )

    # Permute to (M, N, L) layout
    result1 = result1.permute(1, 2, 0)
    result2 = result2.permute(1, 2, 0)

    # Fused SiLU + multiply: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    output = (torch.nn.functional.silu(result1) * result2).to(OUTPUT_DTYPE)

    return output


def custom_kernel_fused(data: input_t) -> output_t:
    """
    More fused version - reuses scale_a across both GEMMs.
    """
    a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu, sfa_perm, sfb1_perm, sfb2_perm, c_out = data

    m, n, l = c_out.shape

    # Output tensor in final layout
    output = torch.empty((m, n, l), dtype=OUTPUT_DTYPE, device="cuda")

    for l_idx in range(l):
        # Convert scale factors once
        scale_a = to_blocked(sfa_cpu[:, :, l_idx]).cuda()
        scale_b1 = to_blocked(sfb1_cpu[:, :, l_idx]).cuda()
        scale_b2 = to_blocked(sfb2_cpu[:, :, l_idx]).cuda()

        a_slice = a[:, :, l_idx]
        b1_t = b1[:, :, l_idx].transpose(0, 1)
        b2_t = b2[:, :, l_idx].transpose(0, 1)

        # Both GEMMs
        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=ACCUMULATOR_DTYPE)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=ACCUMULATOR_DTYPE)

        # Fused epilogue directly to output slice
        output[:, :, l_idx] = (torch.nn.functional.silu(r1) * r2).to(OUTPUT_DTYPE)

    return output
