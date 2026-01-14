#!/usr/bin/env python3
"""
=============================================================================
Step 3 Test: FLOP Calculation and Contiguity Checks
=============================================================================
This test validates:
1. FLOP calculation is mathematically correct (was 1000x off!)
2. Memory calculation is accurate
3. Contiguity checks catch non-contiguous tensors

ACCEPTANCE GATES:
1. FLOP function passes unit tests with hand-computed expected values
2. Non-contiguous inputs fail immediately with clear error
3. Roofline metrics use correct formulas
=============================================================================
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import pytest


# =============================================================================
# Test 1: FLOP Calculation Mathematical Correctness
# =============================================================================
def test_flop_calculation_basic():
    """
    Test FLOP calculation with hand-computed expected values.

    Operation: C = silu(A @ B1) * (A @ B2)

    For single GEMM (M, N, K):
        FLOPs = 2 * M * N * K (one multiply + one add per output element per K)

    For dual GEMM + SiLU + multiply:
        GEMM1: 2 * M * N * K
        GEMM2: 2 * M * N * K
        SiLU:  3 * M * N (exp, add, divide)
        Mul:   1 * M * N
        Total: 4*M*N*K + 4*M*N
    """
    from utils import compute_flops

    print("\n[TEST] FLOP Calculation Basic")

    # Simple case: M=N=K=2, L=1
    # GEMM1: 2*2*2*2 = 16
    # GEMM2: 2*2*2*2 = 16
    # SiLU:  3*2*2 = 12
    # Mul:   1*2*2 = 4
    # Total: 16 + 16 + 12 + 4 = 48
    result = compute_flops(m=2, n=2, k=2, l=1)
    expected = 48
    assert result["total_flops"] == expected, (
        f"FLOP calculation wrong for 2x2x2: got {result['total_flops']}, expected {expected}"
    )
    print(f"  M=2, N=2, K=2, L=1: {result['total_flops']} FLOPs (expected {expected}) [PASS]")

    # Medium case: M=128, N=128, K=256, L=1
    # GEMM1: 2*128*128*256 = 8,388,608
    # GEMM2: 2*128*128*256 = 8,388,608
    # SiLU:  3*128*128 = 49,152
    # Mul:   1*128*128 = 16,384
    # Total: 8,388,608 + 8,388,608 + 49,152 + 16,384 = 16,842,752
    result = compute_flops(m=128, n=128, k=256, l=1)
    expected = 2*128*128*256 + 2*128*128*256 + 3*128*128 + 1*128*128
    assert result["total_flops"] == expected, (
        f"FLOP calculation wrong: got {result['total_flops']}, expected {expected}"
    )
    print(f"  M=128, N=128, K=256, L=1: {result['total_flops']:,} FLOPs [PASS]")

    # Batch case: L=4
    result_batch = compute_flops(m=128, n=128, k=256, l=4)
    assert result_batch["total_flops"] == expected * 4, "Batch scaling wrong"
    print(f"  M=128, N=128, K=256, L=4: {result_batch['total_flops']:,} FLOPs [PASS]")

    print("[PASS] Basic FLOP calculation correct")


def test_flop_calculation_benchmark_sizes():
    """
    Test FLOP calculation for actual benchmark sizes from task.md.

    CRITICAL: Previous bug had these as TFLOP when they should be GFLOP!
    """
    from utils import compute_flops

    print("\n[TEST] FLOP Calculation for Benchmark Sizes")

    benchmark_cases = [
        # (M, N, K, L)
        (256, 4096, 7168, 1),
        (512, 4096, 7168, 1),
        (256, 3072, 4096, 1),
        (512, 3072, 7168, 1),
    ]

    for m, n, k, l in benchmark_cases:
        result = compute_flops(m, n, k, l)

        # Hand-verify the formula
        expected_gemm = 2 * m * n * k  # per GEMM
        expected_silu = 3 * m * n
        expected_mul = m * n
        expected_total = 2 * expected_gemm + expected_silu + expected_mul

        assert result["total_flops"] == expected_total, (
            f"FLOP mismatch for {m}x{n}x{k}: got {result['total_flops']}, expected {expected_total}"
        )

        # Verify GFLOP is in reasonable range (not TFLOP!)
        gflops = result["total_gflops"]
        assert 1 < gflops < 1000, (
            f"GFLOP value {gflops} seems wrong for {m}x{n}x{k} - "
            f"should be in single/double digit GFLOP range, not TFLOP"
        )

        print(f"  {m}x{n}x{k}x{l}: {gflops:.2f} GFLOP (total: {result['total_flops']:,}) [PASS]")

    print("[PASS] Benchmark FLOP calculations correct")


def test_flop_units_not_confused():
    """
    Verify we don't confuse GFLOP with TFLOP.

    The previous bug claimed 60.1 TFLOP for 512x4096x7168 when it's ~60 GFLOP.
    """
    from utils import compute_flops

    print("\n[TEST] FLOP Units Not Confused")

    # Case from gap6_nsight_proof.md that had the 1000x error
    result = compute_flops(m=512, n=4096, k=7168, l=1)

    gflops = result["total_gflops"]
    tflops = result["total_tflops"]

    # Should be ~60 GFLOP, NOT 60 TFLOP
    assert 50 < gflops < 70, f"Expected ~60 GFLOP, got {gflops:.2f}"
    assert 0.05 < tflops < 0.07, f"Expected ~0.06 TFLOP, got {tflops:.4f}"

    print(f"  512x4096x7168: {gflops:.2f} GFLOP = {tflops:.4f} TFLOP")
    print(f"  (Previous bug said 60.1 TFLOP - that was 1000x wrong!)")
    print("[PASS] FLOP units correct")


# =============================================================================
# Test 2: Memory Calculation
# =============================================================================
def test_memory_calculation():
    """
    Test memory calculation accuracy.

    Previous bug said ~33 MB for 512x4096x7168, actual is ~39.3 MB.
    """
    from utils import compute_memory_bytes

    print("\n[TEST] Memory Calculation")

    # Test case from gap6
    result = compute_memory_bytes(m=512, n=4096, k=7168, l=1)

    # Hand-calculate expected values
    sf_k = 7168 // 16  # 448

    expected_a = (512 * 7168) // 2  # FP4 packed
    expected_b1 = (4096 * 7168) // 2
    expected_b2 = (4096 * 7168) // 2
    expected_sfa = 512 * sf_k
    expected_sfb1 = 4096 * sf_k
    expected_sfb2 = 4096 * sf_k
    expected_c = 512 * 4096 * 2  # FP16

    expected_total = (expected_a + expected_b1 + expected_b2 +
                      expected_sfa + expected_sfb1 + expected_sfb2 + expected_c)

    assert result["a_bytes"] == expected_a, f"A bytes wrong: {result['a_bytes']} vs {expected_a}"
    assert result["b1_bytes"] == expected_b1, f"B1 bytes wrong"
    assert result["b2_bytes"] == expected_b2, f"B2 bytes wrong"
    assert result["total_bytes"] == expected_total, f"Total bytes wrong"

    print(f"  A: {result['a_bytes']:,} bytes ({result['a_bytes']/1024/1024:.2f} MB)")
    print(f"  B1: {result['b1_bytes']:,} bytes ({result['b1_bytes']/1024/1024:.2f} MB)")
    print(f"  B2: {result['b2_bytes']:,} bytes ({result['b2_bytes']/1024/1024:.2f} MB)")
    print(f"  Scale factors: {result['sfa_bytes'] + result['sfb1_bytes'] + result['sfb2_bytes']:,} bytes")
    print(f"  Output C: {result['c_bytes']:,} bytes ({result['c_bytes']/1024/1024:.2f} MB)")
    print(f"  Total: {result['total_mb']:.2f} MB")
    print(f"  (Previous bug said ~33 MB, actual is ~{result['total_mb']:.1f} MB)")

    print("[PASS] Memory calculation correct")


# =============================================================================
# Test 3: Arithmetic Intensity
# =============================================================================
def test_arithmetic_intensity():
    """
    Test arithmetic intensity calculation for roofline analysis.
    """
    from utils import compute_arithmetic_intensity

    print("\n[TEST] Arithmetic Intensity")

    result = compute_arithmetic_intensity(m=512, n=4096, k=7168, l=1)

    ai = result["arithmetic_intensity"]

    # Should be ~1.5 ops/byte (NOT ~1800 as the bug claimed)
    assert 1.0 < ai < 2.0, f"AI should be ~1.5, got {ai:.2f}"

    print(f"  512x4096x7168:")
    print(f"    FLOPs: {result['total_gflops']:.2f} GFLOP")
    print(f"    Bytes: {result['total_mb']:.2f} MB")
    print(f"    Arithmetic Intensity: {ai:.2f} ops/byte")
    print(f"  (Previous bug said AI=1800 - that was 1000x wrong!)")

    print("[PASS] Arithmetic intensity correct")


# =============================================================================
# Test 4: Contiguity Validation
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_contiguity_validation():
    """
    Verify contiguity checks catch non-contiguous tensors.
    """
    from constants import assert_contiguous

    print("\n[TEST] Contiguity Validation")

    # Create contiguous tensor - should pass
    t_contig = torch.randn(10, 20, device='cuda')
    assert_contiguous(t_contig, "contiguous_tensor")
    print("  [PASS] Contiguous tensor accepted")

    # Create non-contiguous tensor (transpose without contiguous)
    t_noncontig = torch.randn(20, 10, device='cuda').t()  # Not contiguous!
    assert not t_noncontig.is_contiguous(), "Test setup failed - tensor should be non-contiguous"

    try:
        assert_contiguous(t_noncontig, "non_contiguous_tensor")
        pytest.fail("Should have raised ValueError for non-contiguous tensor")
    except ValueError as e:
        print(f"  [PASS] Non-contiguous tensor rejected: {str(e)[:50]}...")

    # Create non-contiguous via slicing
    t_sliced = torch.randn(10, 20, device='cuda')[:, ::2]  # Non-contiguous slice
    assert not t_sliced.is_contiguous()

    try:
        assert_contiguous(t_sliced, "sliced_tensor")
        pytest.fail("Should have raised ValueError for sliced tensor")
    except ValueError as e:
        print(f"  [PASS] Sliced tensor rejected: {str(e)[:50]}...")

    print("[PASS] Contiguity validation working correctly")


# =============================================================================
# Test 5: Full Input Validation
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_validate_all_inputs():
    """
    Verify validate_all_inputs catches all problems.
    """
    from task import generate_input
    from constants import validate_all_inputs

    print("\n[TEST] Full Input Validation")

    # Generate valid inputs
    m, n, k, l = 128, 128, 256, 1
    inputs = generate_input(m, n, k, l, seed=42)

    # Should pass with valid inputs
    try:
        validate_all_inputs(*inputs)
        print("  [PASS] Valid inputs accepted")
    except Exception as e:
        pytest.fail(f"Valid inputs should be accepted: {e}")

    # Test with non-contiguous output
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c = inputs

    # Make c non-contiguous
    c_big = torch.randn(m * 2, n, l, dtype=c.dtype, device=c.device)
    c_noncontig = c_big[::2, :, :]  # Non-contiguous slice
    assert not c_noncontig.is_contiguous()

    bad_inputs = (a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c_noncontig)

    try:
        validate_all_inputs(*bad_inputs)
        pytest.fail("Should reject non-contiguous output")
    except ValueError as e:
        print(f"  [PASS] Non-contiguous output rejected: {str(e)[:50]}...")

    print("[PASS] Full input validation working")


# =============================================================================
# Test 6: Dimension Constraints
# =============================================================================
def test_dimension_constraints():
    """
    Verify dimension constraints are enforced.
    """
    from constants import assert_dimensions, K_DIVISIBILITY

    print("\n[TEST] Dimension Constraints")

    # Valid dimensions
    assert_dimensions(m=128, n=128, k=256, l=1)
    print(f"  [PASS] Valid dimensions accepted (K=256)")

    assert_dimensions(m=256, n=4096, k=7168, l=1)
    print(f"  [PASS] Benchmark dimensions accepted (K=7168)")

    # Invalid K (not divisible by 256)
    try:
        assert_dimensions(m=128, n=128, k=128, l=1)  # K=128 not divisible by 256
        pytest.fail("Should reject K=128")
    except ValueError as e:
        print(f"  [PASS] K=128 rejected (not divisible by {K_DIVISIBILITY})")

    try:
        assert_dimensions(m=128, n=128, k=192, l=1)  # K=192 not divisible by 256
        pytest.fail("Should reject K=192")
    except ValueError as e:
        print(f"  [PASS] K=192 rejected")

    print("[PASS] Dimension constraints enforced")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 3 TEST: FLOP CALCULATION AND CONTIGUITY CHECKS")
    print("="*70)

    test_flop_calculation_basic()
    test_flop_calculation_benchmark_sizes()
    test_flop_units_not_confused()
    test_memory_calculation()
    test_arithmetic_intensity()
    test_dimension_constraints()

    if torch.cuda.is_available():
        test_contiguity_validation()
        test_validate_all_inputs()
    else:
        print("\n[SKIP] CUDA tests (no GPU available)")

    print("\n" + "="*70)
    print("STEP 3 ACCEPTANCE GATE: PASSED")
    print("="*70 + "\n")
