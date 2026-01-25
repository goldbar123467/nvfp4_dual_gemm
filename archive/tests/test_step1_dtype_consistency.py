#!/usr/bin/env python3
"""
=============================================================================
Step 1 Test: Scale Factor Dtype Consistency
=============================================================================
This test validates that scale factor dtype is consistent everywhere.

ACCEPTANCE GATES:
1. All scale factor tensors have correct dtype (SCALE_FACTOR_DTYPE)
2. Attempting to use wrong dtype raises clear error
3. Reference kernel only accepts correct dtype
=============================================================================
"""

import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import pytest
from typing import Tuple


# =============================================================================
# Test 1: Verify SCALE_FACTOR_DTYPE is correctly defined
# =============================================================================
def test_scale_factor_dtype_defined():
    """Verify the single source of truth dtype constant is defined."""
    from constants import SCALE_FACTOR_DTYPE, SCALE_FACTOR_DTYPE_NAME

    # Must be a valid torch dtype
    assert hasattr(torch, 'float8_e4m3fn'), "PyTorch must support float8_e4m3fn"

    # Must match expected value
    assert SCALE_FACTOR_DTYPE == torch.float8_e4m3fn, (
        f"SCALE_FACTOR_DTYPE is {SCALE_FACTOR_DTYPE}, expected torch.float8_e4m3fn. "
        f"This dtype must match the reference implementation for validation to pass."
    )

    print(f"[PASS] SCALE_FACTOR_DTYPE = {SCALE_FACTOR_DTYPE_NAME}")


# =============================================================================
# Test 2: Verify assert_scale_dtype catches wrong dtype
# =============================================================================
def test_assert_scale_dtype_catches_wrong_dtype():
    """Verify that using wrong dtype raises clear error."""
    from constants import assert_scale_dtype, SCALE_FACTOR_DTYPE

    # Create tensor with CORRECT dtype - should not raise
    correct_tensor = torch.zeros(10, dtype=SCALE_FACTOR_DTYPE, device='cuda')
    try:
        assert_scale_dtype(correct_tensor, "test_correct")
        print("[PASS] Correct dtype accepted")
    except TypeError as e:
        pytest.fail(f"Correct dtype should not raise: {e}")

    # Create tensor with WRONG dtype - should raise
    wrong_dtypes = [
        torch.float32,
        torch.float16,
        torch.int8,
    ]

    # Check if e4m3fnuz exists and add it to wrong dtypes
    if hasattr(torch, 'float8_e4m3fnuz'):
        wrong_dtypes.append(torch.float8_e4m3fnuz)

    for wrong_dtype in wrong_dtypes:
        wrong_tensor = torch.zeros(10, dtype=wrong_dtype, device='cuda')
        try:
            assert_scale_dtype(wrong_tensor, "test_wrong")
            pytest.fail(f"Should have raised TypeError for dtype {wrong_dtype}")
        except TypeError as e:
            print(f"[PASS] Wrong dtype {wrong_dtype} correctly rejected: {str(e)[:60]}...")


# =============================================================================
# Test 3: Verify generate_input produces correct dtypes
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generate_input_dtype_consistency():
    """Verify all generated tensors have correct dtypes."""
    from task import generate_input
    from constants import SCALE_FACTOR_DTYPE, FP4_PACKED_DTYPE, OUTPUT_DTYPE

    # Use small test dimensions
    m, n, k, l = 128, 128, 256, 1

    inputs = generate_input(m, n, k, l, seed=42)

    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c = inputs

    # Check FP4 tensors
    assert a.dtype == FP4_PACKED_DTYPE, f"a has dtype {a.dtype}, expected {FP4_PACKED_DTYPE}"
    assert b1.dtype == FP4_PACKED_DTYPE, f"b1 has dtype {b1.dtype}, expected {FP4_PACKED_DTYPE}"
    assert b2.dtype == FP4_PACKED_DTYPE, f"b2 has dtype {b2.dtype}, expected {FP4_PACKED_DTYPE}"
    print("[PASS] FP4 tensors have correct dtype")

    # Check scale factors (reference format)
    assert sfa.dtype == SCALE_FACTOR_DTYPE, f"sfa has dtype {sfa.dtype}, expected {SCALE_FACTOR_DTYPE}"
    assert sfb1.dtype == SCALE_FACTOR_DTYPE, f"sfb1 has dtype {sfb1.dtype}, expected {SCALE_FACTOR_DTYPE}"
    assert sfb2.dtype == SCALE_FACTOR_DTYPE, f"sfb2 has dtype {sfb2.dtype}, expected {SCALE_FACTOR_DTYPE}"
    print("[PASS] Reference scale factors have correct dtype")

    # Check permuted scale factors
    assert sfa_perm.dtype == SCALE_FACTOR_DTYPE, f"sfa_perm has dtype {sfa_perm.dtype}, expected {SCALE_FACTOR_DTYPE}"
    assert sfb1_perm.dtype == SCALE_FACTOR_DTYPE, f"sfb1_perm has dtype {sfb1_perm.dtype}, expected {SCALE_FACTOR_DTYPE}"
    assert sfb2_perm.dtype == SCALE_FACTOR_DTYPE, f"sfb2_perm has dtype {sfb2_perm.dtype}, expected {SCALE_FACTOR_DTYPE}"
    print("[PASS] Permuted scale factors have correct dtype")

    # Check output
    assert c.dtype == OUTPUT_DTYPE, f"c has dtype {c.dtype}, expected {OUTPUT_DTYPE}"
    print("[PASS] Output tensor has correct dtype")


# =============================================================================
# Test 4: Verify runtime assertion catches dtype mismatch in ref_kernel
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_ref_kernel_dtype_validation():
    """Verify ref_kernel validates input dtypes at runtime."""
    from task import generate_input, ref_kernel
    from constants import SCALE_FACTOR_DTYPE

    # Generate valid inputs
    m, n, k, l = 128, 128, 256, 1
    inputs = generate_input(m, n, k, l, seed=42)

    # Running with correct dtypes should work
    try:
        output = ref_kernel(inputs)
        print("[PASS] ref_kernel accepts correct dtypes")
    except Exception as e:
        pytest.fail(f"ref_kernel should accept correct dtypes: {e}")

    # Create inputs with wrong scale factor dtype
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c = inputs

    # Replace sfa with wrong dtype
    wrong_sfa = sfa.to(torch.float32)
    wrong_inputs = (a, b1, b2, wrong_sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c)

    try:
        output = ref_kernel(wrong_inputs)
        pytest.fail("ref_kernel should reject wrong dtype for sfa")
    except (TypeError, RuntimeError) as e:
        print(f"[PASS] ref_kernel correctly rejects wrong dtype: {str(e)[:60]}...")


# =============================================================================
# Test 5: Document the dtype decision
# =============================================================================
def test_dtype_documentation():
    """Print dtype documentation for clarity."""
    from constants import (
        SCALE_FACTOR_DTYPE,
        SCALE_FACTOR_DTYPE_NAME,
        FP4_PACKED_DTYPE,
        FP4_PACKED_DTYPE_NAME,
        OUTPUT_DTYPE,
        OUTPUT_DTYPE_NAME,
    )

    print("\n" + "="*70)
    print("NVFP4 DUAL-GEMM DATA TYPE SPECIFICATION")
    print("="*70)
    print(f"Scale Factor dtype: {SCALE_FACTOR_DTYPE_NAME}")
    print(f"  - This is the SINGLE SOURCE OF TRUTH")
    print(f"  - Matches reference implementation (task.py ref_kernel)")
    print(f"  - Note: Spec doc mentioned 'e4m3fnuz' but code uses 'e4m3fn'")
    print(f"  - Using e4m3fn to match validation ground truth")
    print()
    print(f"FP4 Packed dtype: {FP4_PACKED_DTYPE_NAME}")
    print(f"  - 2 FP4 values per byte")
    print(f"  - Low nibble = first value, High nibble = second value")
    print()
    print(f"Output dtype: {OUTPUT_DTYPE_NAME}")
    print("="*70 + "\n")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 1 TEST: SCALE FACTOR DTYPE CONSISTENCY")
    print("="*70 + "\n")

    test_scale_factor_dtype_defined()
    test_assert_scale_dtype_catches_wrong_dtype()

    if torch.cuda.is_available():
        test_generate_input_dtype_consistency()
        test_ref_kernel_dtype_validation()
    else:
        print("[SKIP] CUDA tests (no GPU available)")

    test_dtype_documentation()

    print("\n" + "="*70)
    print("STEP 1 ACCEPTANCE GATE: PASSED")
    print("="*70 + "\n")
