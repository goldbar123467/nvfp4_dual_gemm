# =============================================================================
# NVFP4 Dual-GEMM Utility Functions
# =============================================================================

import torch
from typing import Callable, Tuple, Any

# Import from single source of truth
from constants import RTOL, ATOL, SCALE_FACTOR_DTYPE, SCALE_FACTOR_DTYPE_NAME


def ceil_div(a: int, b: int) -> int:
    """Ceiling division: ceil(a / b)"""
    return (a + b - 1) // b


def make_match_reference(
    ref_fn: Callable,
    rtol: float = RTOL,
    atol: float = ATOL
) -> Callable:
    """
    Create a validation function that checks kernel output against reference.

    Args:
        ref_fn: Reference kernel function
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Validation function that returns (passed, error_info)
    """
    def check_implementation(
        kernel_fn: Callable,
        inputs: Tuple[Any, ...],
    ) -> Tuple[bool, dict]:
        """
        Check if kernel output matches reference.

        Returns:
            (passed, error_info) where error_info contains diagnostic data
        """
        # Get reference output
        ref_output = ref_fn(inputs)

        # Get kernel output
        kernel_output = kernel_fn(inputs)

        # Compute errors
        abs_diff = (kernel_output - ref_output).abs()
        max_abs_err = abs_diff.max().item()
        mean_abs_err = abs_diff.mean().item()

        # Relative error (avoid div by zero)
        ref_abs = ref_output.abs()
        rel_diff = abs_diff / (ref_abs + atol)
        max_rel_err = rel_diff.max().item()

        # Check if within tolerance
        passed = torch.allclose(kernel_output, ref_output, rtol=rtol, atol=atol)

        error_info = {
            "passed": passed,
            "max_abs_error": max_abs_err,
            "mean_abs_error": mean_abs_err,
            "max_rel_error": max_rel_err,
            "rtol": rtol,
            "atol": atol,
            "output_shape": tuple(kernel_output.shape),
            "output_dtype": str(kernel_output.dtype),
        }

        return passed, error_info

    return check_implementation


# =============================================================================
# FLOP CALCULATION - CRITICAL: Must be mathematically correct
# =============================================================================

def compute_flops(m: int, n: int, k: int, l: int = 1) -> dict:
    """
    Compute FLOPs for dual GEMM with SiLU activation.

    Operation: C = silu(A @ B1) * (A @ B2)

    FLOP breakdown:
    - GEMM1 (A @ B1): 2*M*N*K (multiply-add = 2 ops)
    - GEMM2 (A @ B2): 2*M*N*K (multiply-add = 2 ops)
    - SiLU per element: ~3 ops (exp, add, divide)
    - Element-wise multiply: M*N ops

    Total per batch: 4*M*N*K + 4*M*N

    Args:
        m, n, k: Matrix dimensions
        l: Batch size

    Returns:
        Dictionary with FLOP breakdown
    """
    gemm1_flops = 2 * m * n * k
    gemm2_flops = 2 * m * n * k
    silu_flops = 3 * m * n  # exp(-x), 1+exp(-x), x/(1+exp(-x))
    mul_flops = m * n  # element-wise multiply

    total_per_batch = gemm1_flops + gemm2_flops + silu_flops + mul_flops
    total_flops = total_per_batch * l

    return {
        "gemm1_flops": gemm1_flops * l,
        "gemm2_flops": gemm2_flops * l,
        "silu_flops": silu_flops * l,
        "mul_flops": mul_flops * l,
        "total_flops": total_flops,
        "total_gflops": total_flops / 1e9,
        "total_tflops": total_flops / 1e12,
        "formula": "4*M*N*K + 4*M*N (per batch)",
    }


def compute_memory_bytes(m: int, n: int, k: int, l: int = 1) -> dict:
    """
    Compute memory bytes transferred for dual GEMM.

    Inputs:
    - A: M x K x L in FP4 = M*K*L/2 bytes
    - B1: N x K x L in FP4 = N*K*L/2 bytes
    - B2: N x K x L in FP4 = N*K*L/2 bytes
    - sfa: M x (K/16) x L in FP8 = M*(K/16)*L bytes
    - sfb1: N x (K/16) x L in FP8 = N*(K/16)*L bytes
    - sfb2: N x (K/16) x L in FP8 = N*(K/16)*L bytes

    Output:
    - C: M x N x L in FP16 = M*N*L*2 bytes

    Args:
        m, n, k: Matrix dimensions
        l: Batch size

    Returns:
        Dictionary with memory breakdown
    """
    sf_k = k // 16  # Number of scale factor blocks along K

    a_bytes = (m * k * l) // 2  # FP4 packed
    b1_bytes = (n * k * l) // 2
    b2_bytes = (n * k * l) // 2
    sfa_bytes = m * sf_k * l  # FP8
    sfb1_bytes = n * sf_k * l
    sfb2_bytes = n * sf_k * l
    c_bytes = m * n * l * 2  # FP16

    total_input_bytes = a_bytes + b1_bytes + b2_bytes + sfa_bytes + sfb1_bytes + sfb2_bytes
    total_bytes = total_input_bytes + c_bytes

    return {
        "a_bytes": a_bytes,
        "b1_bytes": b1_bytes,
        "b2_bytes": b2_bytes,
        "sfa_bytes": sfa_bytes,
        "sfb1_bytes": sfb1_bytes,
        "sfb2_bytes": sfb2_bytes,
        "c_bytes": c_bytes,
        "total_input_bytes": total_input_bytes,
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
    }


def compute_arithmetic_intensity(m: int, n: int, k: int, l: int = 1) -> dict:
    """
    Compute arithmetic intensity (ops/byte) for roofline analysis.

    Args:
        m, n, k: Matrix dimensions
        l: Batch size

    Returns:
        Dictionary with roofline metrics
    """
    flops = compute_flops(m, n, k, l)
    memory = compute_memory_bytes(m, n, k, l)

    ai = flops["total_flops"] / memory["total_bytes"]

    return {
        "arithmetic_intensity": ai,
        "total_flops": flops["total_flops"],
        "total_bytes": memory["total_bytes"],
        "total_gflops": flops["total_gflops"],
        "total_mb": memory["total_mb"],
    }
