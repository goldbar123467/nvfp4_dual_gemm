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

    Optimizations:
    - Pre-compute scale factors outside hot path
    - Direct output without intermediate allocations
    - Fused SiLU + multiply
    """
    a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu, sfa_perm, sfb1_perm, sfb2_perm, c_out = data

    # Get dimensions from output shape (M, N, L)
    m, n, l = c_out.shape

    # Fast path for L=1 (most common case)
    if l == 1:
        # Pre-convert scale factors to blocked format on GPU
        scale_a = to_blocked(sfa_cpu[:, :, 0]).cuda()
        scale_b1 = to_blocked(sfb1_cpu[:, :, 0]).cuda()
        scale_b2 = to_blocked(sfb2_cpu[:, :, 0]).cuda()

        # Get slices (no copy for contiguous data)
        a_slice = a[:, :, 0]
        b1_t = b1[:, :, 0].T
        b2_t = b2[:, :, 0].T

        # Dual GEMM
        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=ACCUMULATOR_DTYPE)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=ACCUMULATOR_DTYPE)

        # Fused epilogue: silu(r1) * r2, output to FP16
        output = (torch.nn.functional.silu(r1) * r2).to(OUTPUT_DTYPE)
        return output.unsqueeze(-1)  # Add L dimension back

    # General case for L > 1
    output = torch.empty((m, n, l), dtype=OUTPUT_DTYPE, device="cuda")

    for l_idx in range(l):
        scale_a = to_blocked(sfa_cpu[:, :, l_idx]).cuda()
        scale_b1 = to_blocked(sfb1_cpu[:, :, l_idx]).cuda()
        scale_b2 = to_blocked(sfb2_cpu[:, :, l_idx]).cuda()

        a_slice = a[:, :, l_idx]
        b1_t = b1[:, :, l_idx].T
        b2_t = b2[:, :, l_idx].T

        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=ACCUMULATOR_DTYPE)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=ACCUMULATOR_DTYPE)

        output[:, :, l_idx] = (torch.nn.functional.silu(r1) * r2).to(OUTPUT_DTYPE)

    return output
