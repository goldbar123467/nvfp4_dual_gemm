# =============================================================================
# NVFP4 Dual-GEMM Submission
# =============================================================================
# GPU MODE Leaderboard Submission
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
# =============================================================================

import torch
from typing import Tuple

# -----------------------------------------------------------------------------
# Constants (inlined for self-contained submission)
# -----------------------------------------------------------------------------
SCALE_FACTOR_DTYPE = torch.float8_e4m3fn
OUTPUT_DTYPE = torch.float16
ACCUMULATOR_DTYPE = torch.float32
SCALE_FACTOR_BLOCK_SIZE = 16
ATOM_M = (32, 4)
ATOM_K = 4

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
    torch.Tensor,  # sfa_permuted
    torch.Tensor,  # sfb1_permuted
    torch.Tensor,  # sfb2_permuted
    torch.Tensor,  # c: [M, N, L] FP16 output
]
output_t = torch.Tensor

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """Convert scale factor tensor to cuBLAS blocked format."""
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


# -----------------------------------------------------------------------------
# Main kernel function
# -----------------------------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    """
    NVFP4 block-scaled dual GEMM with SiLU activation.

    Computes: C = silu(A @ B1) * (A @ B2)

    Optimized for NVIDIA B200 using torch._scaled_mm for FP4 tensor cores.
    """
    a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu, sfa_perm, sfb1_perm, sfb2_perm, c_out = data

    # Get dimensions from output shape (M, N, L)
    m, n, l = c_out.shape

    # Fast path for L=1 (most common case)
    if l == 1:
        scale_a = to_blocked(sfa_cpu[:, :, 0]).cuda()
        scale_b1 = to_blocked(sfb1_cpu[:, :, 0]).cuda()
        scale_b2 = to_blocked(sfb2_cpu[:, :, 0]).cuda()

        a_slice = a[:, :, 0]
        b1_t = b1[:, :, 0].T
        b2_t = b2[:, :, 0].T

        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=ACCUMULATOR_DTYPE)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=ACCUMULATOR_DTYPE)

        output = (torch.nn.functional.silu(r1) * r2).to(OUTPUT_DTYPE)
        return output.unsqueeze(-1)

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
