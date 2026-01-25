# =============================================================================
# NVFP4 Dual-GEMM Submission with Fused Epilogue
# =============================================================================
# Uses custom CUDA kernel for fused SiLU * multiply
# Build first: cd src && pip install -e .
# =============================================================================

import torch
from typing import Tuple

# Try to import the fused kernel, fall back to PyTorch if not built
try:
    import silu_mul_cuda
    USE_FUSED = True
except ImportError:
    USE_FUSED = False
    print("Warning: silu_mul_cuda not found, using PyTorch fallback")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
OUTPUT_DTYPE = torch.float16
ACCUMULATOR_DTYPE = torch.float32

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

# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------
def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    blocks = input_matrix.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.flatten()

# -----------------------------------------------------------------------------
# Main kernel
# -----------------------------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, sfa_cpu, sfb1_cpu, sfb2_cpu, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape

    if l == 1:
        # Convert scale factors
        scale_a = to_blocked(sfa_cpu[:, :, 0]).cuda()
        scale_b1 = to_blocked(sfb1_cpu[:, :, 0]).cuda()
        scale_b2 = to_blocked(sfb2_cpu[:, :, 0]).cuda()

        a_slice = a[:, :, 0]
        b1_t = b1[:, :, 0].T
        b2_t = b2[:, :, 0].T

        # Compute both GEMMs
        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=ACCUMULATOR_DTYPE)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=ACCUMULATOR_DTYPE)

        # Fused epilogue
        if USE_FUSED:
            output = silu_mul_cuda.silu_mul(r1, r2)
        else:
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

        if USE_FUSED:
            output[:, :, l_idx] = silu_mul_cuda.silu_mul(r1, r2)
        else:
            output[:, :, l_idx] = (torch.nn.functional.silu(r1) * r2).to(OUTPUT_DTYPE)

    return output
