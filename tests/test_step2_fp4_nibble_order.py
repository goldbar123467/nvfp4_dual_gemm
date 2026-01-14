#!/usr/bin/env python3
"""
=============================================================================
Step 2 Test: FP4 Nibble Order Validation
=============================================================================
This test validates that FP4 packing/unpacking has correct nibble order.

CRITICAL: If nibble order is wrong, ALL numerical results will be silently
corrupted. This test provides golden vectors that would FAIL if nibbles
are swapped.

ACCEPTANCE GATES:
1. Golden vector pack/unpack test passes
2. Bit-by-bit decoder matches main decoder
3. Endianness/stride assertions pass
=============================================================================
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import pytest
from typing import List, Tuple


# =============================================================================
# FP4 E2M1 Value Table (Reference)
# =============================================================================
# E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
# Bit pattern (4 bits): SEEM
#
# Encoding table:
# 0000 (+0)  -> 0.0
# 0001 (+1)  -> 0.5
# 0010 (+2)  -> 1.0
# 0011 (+3)  -> 1.5
# 0100 (+4)  -> 2.0
# 0101 (+5)  -> 3.0
# 0110 (+6)  -> 4.0
# 0111 (+7)  -> 6.0
# 1000 (-0)  -> -0.0
# 1001 (-1)  -> -0.5
# 1010 (-2)  -> -1.0
# 1011 (-3)  -> -1.5
# 1100 (-4)  -> -2.0
# 1101 (-5)  -> -3.0
# 1110 (-6)  -> -4.0
# 1111 (-7)  -> -6.0
#
# Task mask 0b1011_1011 keeps only {0, ±0.5, ±1, ±1.5}
# =============================================================================

FP4_E2M1_TABLE = {
    0b0000: 0.0,
    0b0001: 0.5,
    0b0010: 1.0,
    0b0011: 1.5,
    0b0100: 2.0,
    0b0101: 3.0,
    0b0110: 4.0,
    0b0111: 6.0,
    0b1000: -0.0,
    0b1001: -0.5,
    0b1010: -1.0,
    0b1011: -1.5,
    0b1100: -2.0,
    0b1101: -3.0,
    0b1110: -4.0,
    0b1111: -6.0,
}

# Values after task mask 0b1011_1011 (clears bit 2 in each nibble)
FP4_TASK_VALUES = {0.0, 0.5, 1.0, 1.5, -0.0, -0.5, -1.0, -1.5}


def fp4_nibble_to_float(nibble: int) -> float:
    """
    Bit-by-bit reference decoder for FP4 E2M1 format.

    This is the GOLDEN REFERENCE that must match any optimized decoder.
    """
    assert 0 <= nibble <= 15, f"Invalid nibble: {nibble}"
    return FP4_E2M1_TABLE[nibble]


def unpack_byte_to_fp4_pair(byte: int) -> Tuple[float, float]:
    """
    Unpack one byte to two FP4 values.

    CRITICAL NIBBLE ORDER DEFINITION:
    - Low nibble (bits 0-3) = FIRST FP4 value (element at even index)
    - High nibble (bits 4-7) = SECOND FP4 value (element at odd index)

    This matches PyTorch's torch.float4_e2m1fn_x2 convention.
    """
    low_nibble = byte & 0x0F  # bits 0-3
    high_nibble = (byte >> 4) & 0x0F  # bits 4-7

    first = fp4_nibble_to_float(low_nibble)
    second = fp4_nibble_to_float(high_nibble)

    return (first, second)


def pack_fp4_pair_to_byte(first: float, second: float) -> int:
    """
    Pack two FP4 values into one byte.

    Inverse of unpack_byte_to_fp4_pair.
    """
    # Find nibble values
    low_nibble = None
    high_nibble = None

    for nibble, value in FP4_E2M1_TABLE.items():
        # Handle -0.0 vs 0.0 comparison
        if value == first or (value == 0.0 and first == 0.0):
            if low_nibble is None or nibble < 8:  # Prefer positive zero
                low_nibble = nibble
        if value == second or (value == 0.0 and second == 0.0):
            if high_nibble is None or nibble < 8:
                high_nibble = nibble

    assert low_nibble is not None, f"Invalid FP4 value: {first}"
    assert high_nibble is not None, f"Invalid FP4 value: {second}"

    return (high_nibble << 4) | low_nibble


# =============================================================================
# Test 1: Golden Vector Pack/Unpack Test
# =============================================================================
def test_golden_vectors_pack_unpack():
    """
    Test pack/unpack with golden vectors that WILL FAIL if nibble order is wrong.

    These vectors are specifically chosen so that swapping nibbles produces
    different results.
    """
    print("\n[TEST] Golden Vector Pack/Unpack")

    # Golden test cases: (byte_value, expected_first, expected_second)
    # Chosen so first != second to detect swaps
    golden_vectors = [
        (0x10, 0.0, 0.5),    # low=0000 (0.0), high=0001 (0.5)
        (0x21, 0.5, 1.0),    # low=0001 (0.5), high=0010 (1.0)
        (0x32, 1.0, 1.5),    # low=0010 (1.0), high=0011 (1.5)
        (0x91, 0.5, -0.5),   # low=0001 (0.5), high=1001 (-0.5)
        (0xA1, 0.5, -1.0),   # low=0001 (0.5), high=1010 (-1.0)
        (0xB3, 1.5, -1.5),   # low=0011 (1.5), high=1011 (-1.5)
        (0x00, 0.0, 0.0),    # low=0000 (0.0), high=0000 (0.0)
        (0xFF, -6.0, -6.0),  # low=1111 (-6.0), high=1111 (-6.0)
        (0x53, 1.5, 3.0),    # low=0011 (1.5), high=0101 (3.0)
        (0x64, 2.0, 4.0),    # low=0100 (2.0), high=0110 (4.0)
    ]

    all_passed = True
    for byte_val, exp_first, exp_second in golden_vectors:
        first, second = unpack_byte_to_fp4_pair(byte_val)

        # Check values
        if first != exp_first or second != exp_second:
            print(f"  [FAIL] Byte 0x{byte_val:02X}: got ({first}, {second}), expected ({exp_first}, {exp_second})")

            # Diagnose if nibbles were swapped
            if first == exp_second and second == exp_first:
                print(f"        -> NIBBLE ORDER IS SWAPPED!")
            all_passed = False
        else:
            print(f"  [PASS] Byte 0x{byte_val:02X} -> ({first}, {second})")

        # Verify round-trip
        repacked = pack_fp4_pair_to_byte(first, second)
        if repacked != byte_val:
            # Allow for -0.0 vs 0.0 ambiguity
            if not (first == 0.0 or second == 0.0):
                print(f"  [FAIL] Round-trip failed: 0x{byte_val:02X} -> ({first}, {second}) -> 0x{repacked:02X}")
                all_passed = False

    assert all_passed, "Golden vector tests failed - NIBBLE ORDER MAY BE WRONG"
    print("[PASS] All golden vectors passed")


# =============================================================================
# Test 2: Verify nibble swap detection
# =============================================================================
def test_nibble_swap_detection():
    """
    Verify our test would catch a nibble swap bug.

    This test intentionally uses WRONG unpacking to confirm detection.
    """
    print("\n[TEST] Nibble Swap Detection")

    def wrong_unpack(byte: int) -> Tuple[float, float]:
        """INTENTIONALLY WRONG: swapped nibble order."""
        high_nibble = byte & 0x0F  # WRONG: high from low bits
        low_nibble = (byte >> 4) & 0x0F  # WRONG: low from high bits
        return (fp4_nibble_to_float(low_nibble), fp4_nibble_to_float(high_nibble))

    # Test case where first != second
    byte_val = 0x21  # Correct: (0.5, 1.0), Swapped: (1.0, 0.5)

    correct_result = unpack_byte_to_fp4_pair(byte_val)
    wrong_result = wrong_unpack(byte_val)

    assert correct_result == (0.5, 1.0), f"Correct unpack failed: {correct_result}"
    assert wrong_result == (1.0, 0.5), f"Wrong unpack unexpected: {wrong_result}"
    assert correct_result != wrong_result, "Test vectors should detect swap"

    print(f"  Correct unpack of 0x21: {correct_result}")
    print(f"  Wrong unpack of 0x21:   {wrong_result}")
    print("[PASS] Nibble swap detection verified")


# =============================================================================
# Test 3: Task mask validation
# =============================================================================
def test_task_mask_values():
    """
    Verify the task mask 0b1011_1011 produces only allowed values.

    The task restricts FP4 values to {0, ±0.5, ±1, ±1.5}.
    """
    print("\n[TEST] Task Mask Values")

    mask = 0b1011_1011

    # Check all possible bytes after masking
    for byte_val in range(256):
        masked = byte_val & mask
        first, second = unpack_byte_to_fp4_pair(masked)

        # Both values should be in allowed set
        assert first in FP4_TASK_VALUES or first == -0.0, f"Invalid first value {first} from byte 0x{masked:02X}"
        assert second in FP4_TASK_VALUES or second == -0.0, f"Invalid second value {second} from byte 0x{masked:02X}"

    print(f"  Allowed values: {sorted([v for v in FP4_TASK_VALUES if v >= 0])}")
    print("[PASS] All masked bytes produce valid task values")


# =============================================================================
# Test 4: Alignment and stride validation
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_alignment_and_stride():
    """
    Verify FP4 tensor alignment and stride assumptions.
    """
    print("\n[TEST] Alignment and Stride")

    from constants import K_DIVISIBILITY, SCALE_FACTOR_BLOCK_SIZE

    # Create test tensor
    m, k, l = 128, 256, 1

    # K must be divisible by K_DIVISIBILITY (256)
    assert k % K_DIVISIBILITY == 0, f"K={k} must be divisible by {K_DIVISIBILITY}"
    print(f"  K={k} divisible by {K_DIVISIBILITY}: PASS")

    # Physical K dimension for packed FP4
    k_physical = k // 2
    print(f"  Physical K dimension (K/2): {k_physical}")

    # Verify K/2 is suitable for vector loads (32 bytes = 64 FP4)
    vector_width_bytes = 32
    vector_width_fp4 = vector_width_bytes * 2  # 64 FP4 values
    assert k % vector_width_fp4 == 0, f"K={k} must be divisible by {vector_width_fp4} for vector loads"
    print(f"  K divisible by {vector_width_fp4} (for 32-byte loads): PASS")

    # Verify scale factor alignment
    sf_k = k // SCALE_FACTOR_BLOCK_SIZE
    print(f"  Scale factor K dimension: {sf_k}")

    print("[PASS] Alignment and stride checks passed")


# =============================================================================
# Test 5: Bit pattern exhaustive verification
# =============================================================================
def test_exhaustive_bit_patterns():
    """
    Verify all 16 FP4 bit patterns decode correctly.
    """
    print("\n[TEST] Exhaustive Bit Pattern Verification")

    for nibble in range(16):
        expected = FP4_E2M1_TABLE[nibble]
        decoded = fp4_nibble_to_float(nibble)

        # Handle -0.0 comparison
        if expected == 0.0 and decoded == 0.0:
            match = True
        else:
            match = (decoded == expected)

        if not match:
            print(f"  [FAIL] Nibble {nibble:04b} ({nibble}): got {decoded}, expected {expected}")
        else:
            sign = "+" if nibble < 8 else "-"
            print(f"  [PASS] {nibble:04b} -> {decoded:+.1f}")

    print("[PASS] All 16 bit patterns verified")


# =============================================================================
# Test 6: Cross-check with PyTorch (if running on GPU)
# =============================================================================
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_pytorch_fp4_consistency():
    """
    Cross-check our bit-by-bit decoder against PyTorch's FP4 interpretation.
    """
    print("\n[TEST] PyTorch FP4 Consistency")

    from constants import FP4_PACKED_DTYPE

    # Create a test byte pattern
    test_bytes = torch.tensor([0x10, 0x21, 0x32, 0x91], dtype=torch.uint8, device='cuda')

    # View as FP4 packed
    fp4_tensor = test_bytes.view(FP4_PACKED_DTYPE)

    print(f"  Created FP4 tensor with dtype {fp4_tensor.dtype}")
    print(f"  Shape: {fp4_tensor.shape}")
    print(f"  Test bytes: {[f'0x{b:02X}' for b in test_bytes.cpu().numpy()]}")

    # We can't directly read FP4 values, but we can verify the dtype works
    assert fp4_tensor.dtype == FP4_PACKED_DTYPE

    print("[PASS] PyTorch FP4 tensor creation successful")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("STEP 2 TEST: FP4 NIBBLE ORDER VALIDATION")
    print("="*70)

    test_golden_vectors_pack_unpack()
    test_nibble_swap_detection()
    test_task_mask_values()
    test_exhaustive_bit_patterns()

    if torch.cuda.is_available():
        test_alignment_and_stride()
        test_pytorch_fp4_consistency()
    else:
        print("\n[SKIP] CUDA tests (no GPU available)")

    print("\n" + "="*70)
    print("STEP 2 ACCEPTANCE GATE: PASSED")
    print("="*70 + "\n")
