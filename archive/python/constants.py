# =============================================================================
# NVFP4 Dual-GEMM Project Constants
# =============================================================================
# This file is the SINGLE SOURCE OF TRUTH for all data types and constraints.
# All other code must import from here - do not hardcode dtypes elsewhere.
# =============================================================================

import torch

# -----------------------------------------------------------------------------
# SCALE FACTOR DATA TYPE - CRITICAL CORRECTNESS CONSTRAINT
# -----------------------------------------------------------------------------
# The reference implementation (task.md ref_kernel) uses torch.float8_e4m3fn.
# This is the ground truth for validation. The spec description mentioned
# "e4m3fnuz" but the actual reference code uses e4m3fn.
#
# DECISION: Use torch.float8_e4m3fn to match reference implementation.
# This ensures validation passes against ref_kernel.
#
# Difference between the two formats:
# - e4m3fn: Standard FP8, has +0 and -0, range [-448, 448]
# - e4m3fnuz: Unsigned zero variant, uses 0x80 for NaN, no -0
# -----------------------------------------------------------------------------
SCALE_FACTOR_DTYPE = torch.float8_e4m3fn
SCALE_FACTOR_DTYPE_NAME = "torch.float8_e4m3fn"

# -----------------------------------------------------------------------------
# FP4 DATA TYPE
# -----------------------------------------------------------------------------
# nvfp4 (e2m1) packed format: 2 FP4 values per byte
# Low nibble (bits 0-3) = first FP4 value
# High nibble (bits 4-7) = second FP4 value
# -----------------------------------------------------------------------------
FP4_PACKED_DTYPE = torch.float4_e2m1fn_x2
FP4_PACKED_DTYPE_NAME = "torch.float4_e2m1fn_x2"

# -----------------------------------------------------------------------------
# OUTPUT DATA TYPE
# -----------------------------------------------------------------------------
OUTPUT_DTYPE = torch.float16
OUTPUT_DTYPE_NAME = "torch.float16"

# -----------------------------------------------------------------------------
# ACCUMULATOR DATA TYPE (for internal computation)
# -----------------------------------------------------------------------------
ACCUMULATOR_DTYPE = torch.float32
ACCUMULATOR_DTYPE_NAME = "torch.float32"

# -----------------------------------------------------------------------------
# DIMENSIONAL CONSTRAINTS
# -----------------------------------------------------------------------------
SCALE_FACTOR_BLOCK_SIZE = 16  # Number of FP4 elements per scale factor
K_DIVISIBILITY = 256  # K must be divisible by this
M_ALIGNMENT = 128  # M must be divisible by this for MMA tiling
N_ALIGNMENT = 128  # N must be divisible by this for MMA tiling

# Atom layout dimensions for scale factors
ATOM_M = (32, 4)  # (rows_per_atom, atoms_per_block) -> 128 rows per block
ATOM_K = 4  # Scale blocks per K tile -> 64 elements per K tile

# -----------------------------------------------------------------------------
# VALIDATION TOLERANCES
# -----------------------------------------------------------------------------
RTOL = 1e-03
ATOL = 1e-03

# -----------------------------------------------------------------------------
# ASSERTION HELPERS
# -----------------------------------------------------------------------------
def assert_scale_dtype(tensor: torch.Tensor, name: str = "scale_factor"):
    """Assert that a tensor has the correct scale factor dtype."""
    if tensor.dtype != SCALE_FACTOR_DTYPE:
        raise TypeError(
            f"{name} has dtype {tensor.dtype}, expected {SCALE_FACTOR_DTYPE_NAME}. "
            f"Scale factors must use {SCALE_FACTOR_DTYPE_NAME} to match reference implementation."
        )

def assert_fp4_dtype(tensor: torch.Tensor, name: str = "fp4_tensor"):
    """Assert that a tensor has the correct FP4 packed dtype."""
    if tensor.dtype != FP4_PACKED_DTYPE:
        raise TypeError(
            f"{name} has dtype {tensor.dtype}, expected {FP4_PACKED_DTYPE_NAME}."
        )

def assert_output_dtype(tensor: torch.Tensor, name: str = "output"):
    """Assert that a tensor has the correct output dtype."""
    if tensor.dtype != OUTPUT_DTYPE:
        raise TypeError(
            f"{name} has dtype {tensor.dtype}, expected {OUTPUT_DTYPE_NAME}."
        )

def assert_contiguous(tensor: torch.Tensor, name: str = "tensor"):
    """Assert that a tensor is contiguous in memory."""
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} is not contiguous. All input tensors must be contiguous. "
            f"Use .contiguous() to fix."
        )

def assert_dimensions(m: int, n: int, k: int, l: int):
    """Assert that dimensions meet all constraints."""
    if k % K_DIVISIBILITY != 0:
        raise ValueError(f"K={k} must be divisible by {K_DIVISIBILITY}")
    if k % SCALE_FACTOR_BLOCK_SIZE != 0:
        raise ValueError(f"K={k} must be divisible by scale block size {SCALE_FACTOR_BLOCK_SIZE}")

def validate_all_inputs(a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c):
    """
    Validate ALL input tensors for dtype and contiguity.
    This is a critical correctness gate - call before any kernel execution.
    """
    # Validate FP4 tensors
    assert_fp4_dtype(a, "a")
    assert_fp4_dtype(b1, "b1")
    assert_fp4_dtype(b2, "b2")
    assert_contiguous(a, "a")
    assert_contiguous(b1, "b1")
    assert_contiguous(b2, "b2")

    # Validate scale factors (reference format)
    assert_scale_dtype(sfa, "sfa")
    assert_scale_dtype(sfb1, "sfb1")
    assert_scale_dtype(sfb2, "sfb2")
    assert_contiguous(sfa, "sfa")
    assert_contiguous(sfb1, "sfb1")
    assert_contiguous(sfb2, "sfb2")

    # Validate permuted scale factors
    assert_scale_dtype(sfa_perm, "sfa_permuted")
    assert_scale_dtype(sfb1_perm, "sfb1_permuted")
    assert_scale_dtype(sfb2_perm, "sfb2_permuted")
    assert_contiguous(sfa_perm, "sfa_permuted")
    assert_contiguous(sfb1_perm, "sfb1_permuted")
    assert_contiguous(sfb2_perm, "sfb2_permuted")

    # Validate output
    assert_output_dtype(c, "c")
    assert_contiguous(c, "c")
