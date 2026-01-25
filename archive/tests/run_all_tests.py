#!/usr/bin/env python3
"""
=============================================================================
NVFP4 Dual-GEMM Test Runner
=============================================================================
Runs all validation tests in order:
1. Step 1: Scale factor dtype consistency
2. Step 2: FP4 nibble order validation
3. Step 3: FLOP calculation and contiguity checks

ACCEPTANCE GATES:
All three steps must pass before proceeding to implementation.
=============================================================================
"""

import sys
import os
import importlib.util
from pathlib import Path

# Add python directory to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'python'))


def run_test_module(test_file: Path) -> bool:
    """Run a test module and return success status."""
    print(f"\n{'='*70}")
    print(f"Running: {test_file.name}")
    print('='*70)

    try:
        # Load and run the module
        spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
        module = importlib.util.module_from_spec(spec)

        # Execute the module (which runs tests in __main__)
        spec.loader.exec_module(module)

        return True

    except AssertionError as e:
        print(f"\n[FAILED] AssertionError: {e}")
        return False
    except Exception as e:
        print(f"\n[FAILED] Exception: {type(e).__name__}: {e}")
        return False


def main():
    print("\n" + "="*70)
    print("NVFP4 DUAL-GEMM CORRECTNESS VALIDATION SUITE")
    print("="*70)
    print("\nThis suite validates critical correctness constraints before")
    print("proceeding with kernel implementation. All gates must pass.")

    test_files = [
        SCRIPT_DIR / "test_step1_dtype_consistency.py",
        SCRIPT_DIR / "test_step2_fp4_nibble_order.py",
        SCRIPT_DIR / "test_step3_flops_and_contiguity.py",
    ]

    results = {}
    all_passed = True

    for test_file in test_files:
        if test_file.exists():
            passed = run_test_module(test_file)
            results[test_file.stem] = passed
            if not passed:
                all_passed = False
        else:
            print(f"\n[WARNING] Test file not found: {test_file}")
            results[test_file.stem] = False
            all_passed = False

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {test_name}")

    print("\n" + "="*70)
    if all_passed:
        print("ALL ACCEPTANCE GATES PASSED")
        print("="*70)
        print("\nYou may proceed with kernel implementation.")
        return 0
    else:
        print("SOME GATES FAILED - DO NOT PROCEED")
        print("="*70)
        print("\nFix the failing tests before implementing the kernel.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
