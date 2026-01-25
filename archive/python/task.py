# =============================================================================
# NVFP4 Block Scaled Dual GEMM with SiLU Activation
# =============================================================================
# Reference implementation and input generation for:
#   C = silu(A @ B1) * (A @ B2)
#
# CRITICAL: This file uses constants from constants.py as the single source
# of truth for all data types. Do not hardcode dtypes here.
# =============================================================================

import torch
from typing import Tuple

# Import from single source of truth
from constants import (
    SCALE_FACTOR_DTYPE,
    SCALE_FACTOR_DTYPE_NAME,
    FP4_PACKED_DTYPE,
    OUTPUT_DTYPE,
    ACCUMULATOR_DTYPE,
    SCALE_FACTOR_BLOCK_SIZE,
    K_DIVISIBILITY,
    ATOM_M,
    ATOM_K,
    RTOL,
    ATOL,
    assert_scale_dtype,
    assert_fp4_dtype,
    assert_output_dtype,
    assert_contiguous,
    assert_dimensions,
    validate_all_inputs,
)
from utils import ceil_div, make_match_reference

# Type aliases for documentation
input_t = Tuple[
    torch.Tensor,  # a: [M, K//2, L] FP4 packed
    torch.Tensor,  # b1: [N, K//2, L] FP4 packed
    torch.Tensor,  # b2: [N, K//2, L] FP4 packed
    torch.Tensor,  # sfa: [M, K//16, L] FP8 scale factors (reference format)
    torch.Tensor,  # sfb1: [N, K//16, L] FP8 scale factors (reference format)
    torch.Tensor,  # sfb2: [N, K//16, L] FP8 scale factors (reference format)
    torch.Tensor,  # sfa_permuted: [32, 4, M//128, 4, K//64, L] FP8 scale factors (kernel format)
    torch.Tensor,  # sfb1_permuted: [32, 4, N//128, 4, K//64, L] FP8 scale factors (kernel format)
    torch.Tensor,  # sfb2_permuted: [32, 4, N//128, 4, K//64, L] FP8 scale factors (kernel format)
    torch.Tensor,  # c: [M, N, L] FP16 output
]
output_t = torch.Tensor  # [M, N, L] FP16


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert scale factor tensor to blocked format for torch._scaled_mm.

    This layout is specific to cuBLAS/PyTorch's _scaled_mm operation.

    Args:
        input_matrix: [rows, cols] scale factors

    Returns:
        Flattened blocked representation for _scaled_mm
    """
    rows, cols = input_matrix.shape

    # Ensure dimensions are compatible with blocking
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def ref_kernel(data: input_t) -> output_t:
    """
    PyTorch reference implementation of NVFP4 block-scaled dual GEMM with silu activation.

    Computes: C = silu(A @ B1) * (A @ B2)

    This is the GROUND TRUTH for validation. All kernel implementations must
    produce outputs that match this reference within RTOL/ATOL tolerances.

    Args:
        data: Tuple of input tensors (see input_t type alias)

    Returns:
        Output tensor C in FP16
    """
    a_ref, b1_ref, b2_ref, sfa_ref_cpu, sfb1_ref_cpu, sfb2_ref_cpu, _, _, _, c_ref = data

    # Get dimensions from MxNxL layout
    m, n, l = c_ref.shape

    # Validate scale factor dtypes - CRITICAL correctness check
    assert_scale_dtype(sfa_ref_cpu, "sfa_ref_cpu")
    assert_scale_dtype(sfb1_ref_cpu, "sfb1_ref_cpu")
    assert_scale_dtype(sfb2_ref_cpu, "sfb2_ref_cpu")

    # Allocate FP32 accumulators
    ref1 = torch.empty(
        (l, m, n),
        dtype=ACCUMULATOR_DTYPE,
        device="cuda",
    ).permute(1, 2, 0)

    ref2 = torch.empty(
        (l, m, n),
        dtype=ACCUMULATOR_DTYPE,
        device="cuda",
    ).permute(1, 2, 0)

    # Process each batch
    for l_idx in range(l):
        # Convert scale factors to blocked format for _scaled_mm
        scale_a = to_blocked(sfa_ref_cpu[:, :, l_idx])
        scale_b1 = to_blocked(sfb1_ref_cpu[:, :, l_idx])
        scale_b2 = to_blocked(sfb2_ref_cpu[:, :, l_idx])

        # GEMM1: (m, k) @ (n, k).T -> (m, n)
        res1 = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b1_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b1.cuda(),
            bias=None,
            out_dtype=ACCUMULATOR_DTYPE,
        )
        ref1[:, :, l_idx] = res1

        # GEMM2: (m, k) @ (n, k).T -> (m, n)
        res2 = torch._scaled_mm(
            a_ref[:, :, l_idx],
            b2_ref[:, :, l_idx].transpose(0, 1),
            scale_a.cuda(),
            scale_b2.cuda(),
            bias=None,
            out_dtype=ACCUMULATOR_DTYPE,
        )
        ref2[:, :, l_idx] = res2

    # Fused activation: SiLU on GEMM1 result, multiply with GEMM2 result
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    # All computed in FP32 for numerical stability
    c_ref = (torch.nn.functional.silu(ref1) * ref2).to(OUTPUT_DTYPE)

    return c_ref


def generate_input(
    m: int,
    n: int,
    k: int,
    l: int,
    seed: int,
) -> input_t:
    """
    Generate input tensors for NVFP4 block-scaled dual GEMM with silu activation.

    Args:
        m: Number of rows in matrix A and output C
        n: Number of columns in output C (rows in B)
        k: Inner dimension (columns in A, columns in B before transpose)
        l: Batch size
        seed: Random seed for reproducibility

    Returns:
        Tuple of input tensors:
        - a: [M, K//2, L] - Input matrix in FP4 packed format
        - b1: [N, K//2, L] - Weight matrix 1 in FP4 packed format
        - b2: [N, K//2, L] - Weight matrix 2 in FP4 packed format
        - sfa: [M, K//16, L] - Scale factors for A (reference format)
        - sfb1: [N, K//16, L] - Scale factors for B1 (reference format)
        - sfb2: [N, K//16, L] - Scale factors for B2 (reference format)
        - sfa_permuted: [32, 4, M//128, 4, K//64, L] - Scale factors (kernel format)
        - sfb1_permuted: [32, 4, N//128, 4, K//64, L] - Scale factors (kernel format)
        - sfb2_permuted: [32, 4, N//128, 4, K//64, L] - Scale factors (kernel format)
        - c: [M, N, L] - Output tensor (pre-allocated)

    Data Types (from constants.py - SINGLE SOURCE OF TRUTH):
        - FP4: torch.float4_e2m1fn_x2 (packed)
        - Scale factors: torch.float8_e4m3fn
        - Output: torch.float16
    """
    # Validate dimensional constraints
    assert_dimensions(m, n, k, l)

    torch.manual_seed(seed)

    def create_fp4_tensors(l: int, mn: int, k: int) -> torch.Tensor:
        """Create FP4 packed tensors with valid bit patterns."""
        # Generate uint8 tensor, then convert to float4_e2m1fn_x2
        ref_i8 = torch.randint(255, size=(l, mn, k // 2), dtype=torch.uint8, device="cuda")

        # For each nibble, only keep the sign bit and 2 LSBs
        # Valid values: [-1.5, -1, -0.5, 0, +0.5, +1, +1.5]
        ref_i8 = ref_i8 & 0b1011_1011

        return ref_i8.permute(1, 2, 0).view(FP4_PACKED_DTYPE)

    # Generate FP4 input tensors
    a_ref = create_fp4_tensors(l, m, k)
    b1_ref = create_fp4_tensors(l, n, k)
    b2_ref = create_fp4_tensors(l, n, k)

    # Redundant view to ensure correct dtype (defensive)
    a_ref = a_ref.view(FP4_PACKED_DTYPE)
    b1_ref = b1_ref.view(FP4_PACKED_DTYPE)
    b2_ref = b2_ref.view(FP4_PACKED_DTYPE)

    # Create FP16 output tensor
    c_ref = torch.randn((l, m, n), dtype=OUTPUT_DTYPE, device="cuda").permute(1, 2, 0)

    def create_scale_factor_tensors(l: int, mn: int, sf_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create scale factor tensors in both reference and kernel formats.

        The kernel format (atom layout) is:
            [32, 4, rest_m, 4, rest_k, l]

        This matches the cuBLAS block scaling layout:
            https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

        Returns:
            (reference_tensor, permuted_tensor)
        """
        # Create reference scale factor tensor
        ref_shape = (l, mn, sf_k)
        ref_permute_order = (1, 2, 0)

        # Init with FP32 in [0, 1), then convert to FP8
        # CRITICAL: Use SCALE_FACTOR_DTYPE from constants.py
        ref_f8_random_fp32 = torch.rand(ref_shape, dtype=torch.float32, device='cuda')
        ref_f8_torch_tensor = ref_f8_random_fp32.to(dtype=SCALE_FACTOR_DTYPE)

        # Permute to [mn, sf_k, l]
        ref_f8_torch_tensor_permuted = ref_f8_torch_tensor.permute(*ref_permute_order)

        # Create permuted tensor for kernel
        atom_m = ATOM_M  # (32, 4)
        atom_k = ATOM_K  # 4

        mma_shape = (
            l,
            ceil_div(mn, atom_m[0] * atom_m[1]),  # rest_m
            ceil_div(sf_k, atom_k),  # rest_k
            atom_m[0],  # 32
            atom_m[1],  # 4
            atom_k,  # 4
        )

        # Permute order: (l, rest_m, rest_k, 32, 4, 4) -> (32, 4, rest_m, 4, rest_k, l)
        mma_permute_order = (3, 4, 1, 5, 2, 0)

        # Allocate and convert to FP8
        # CRITICAL: Use SCALE_FACTOR_DTYPE from constants.py
        rand_int_tensor = torch.empty(mma_shape, dtype=torch.int8, device='cuda')
        reordered_f8_torch_tensor = rand_int_tensor.to(dtype=SCALE_FACTOR_DTYPE)
        reordered_f8_torch_tensor = reordered_f8_torch_tensor.permute(*mma_permute_order)

        # GPU-side vectorized reordering
        i_idx = torch.arange(mn, device='cuda')
        j_idx = torch.arange(sf_k, device='cuda')
        b_idx = torch.arange(l, device='cuda')

        i_grid, j_grid, b_grid = torch.meshgrid(i_idx, j_idx, b_idx, indexing='ij')

        # Calculate target indices
        mm = i_grid // (atom_m[0] * atom_m[1])  # Which 128-row block
        mm32 = i_grid % atom_m[0]  # Position within 32-row atom
        mm4 = (i_grid % 128) // atom_m[0]  # Which 32-row atom within 128-block
        kk = j_grid // atom_k  # Which K tile
        kk4 = j_grid % atom_k  # Position within K tile

        # Perform reordering
        reordered_f8_torch_tensor[mm32, mm4, mm, kk4, kk, b_grid] = ref_f8_torch_tensor_permuted[i_grid, j_grid, b_grid]

        return ref_f8_torch_tensor_permuted.cpu(), reordered_f8_torch_tensor

    sf_k = ceil_div(k, SCALE_FACTOR_BLOCK_SIZE)
    sfa_ref_cpu, sfa_ref_permuted = create_scale_factor_tensors(l, m, sf_k)
    sfb1_ref_cpu, sfb1_ref_permuted = create_scale_factor_tensors(l, n, sf_k)
    sfb2_ref_cpu, sfb2_ref_permuted = create_scale_factor_tensors(l, n, sf_k)

    result = (
        a_ref, b1_ref, b2_ref,
        sfa_ref_cpu.to("cuda"), sfb1_ref_cpu.to("cuda"), sfb2_ref_cpu.to("cuda"),
        sfa_ref_permuted, sfb1_ref_permuted, sfb2_ref_permuted,
        c_ref
    )

    # Verify all tensors have correct dtypes before returning
    # This is a CRITICAL correctness gate
    assert_scale_dtype(result[3], "sfa")
    assert_scale_dtype(result[4], "sfb1")
    assert_scale_dtype(result[5], "sfb2")
    assert_scale_dtype(result[6], "sfa_permuted")
    assert_scale_dtype(result[7], "sfb1_permuted")
    assert_scale_dtype(result[8], "sfb2_permuted")

    return result


# Create validation function using reference kernel
check_implementation = make_match_reference(ref_kernel, rtol=RTOL, atol=ATOL)
