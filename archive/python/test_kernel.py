#!/usr/bin/env python3
"""
Test custom kernel against reference implementation.
"""

import torch
import sys
sys.path.insert(0, '/workspace/nvfp4_dual_gemm/python')

from task import generate_input, ref_kernel, check_implementation
from kernel import custom_kernel, custom_kernel_fused


def test_kernel(m, n, k, l=1, seed=42):
    """Test custom kernel against reference for given dimensions."""
    print(f"\n{'='*60}")
    print(f"Testing M={m}, N={n}, K={k}, L={l}")
    print(f"{'='*60}")

    # Generate input data
    data = generate_input(m, n, k, l, seed)

    # Run reference kernel
    print("Running reference kernel...")
    ref_output = ref_kernel(data)

    # Run custom kernel
    print("Running custom kernel...")
    custom_output = custom_kernel(data)

    # Run fused kernel
    print("Running fused kernel...")
    fused_output = custom_kernel_fused(data)

    # Compare outputs
    print("\nResults:")

    # Custom vs Reference
    passed, info = check_implementation(custom_kernel, data)
    print(f"  Custom kernel:  {'PASS' if passed else 'FAIL'}")
    print(f"    Max abs error: {info['max_abs_error']:.6e}")
    print(f"    Max rel error: {info['max_rel_error']:.6e}")

    # Fused vs Reference
    passed_fused, info_fused = check_implementation(custom_kernel_fused, data)
    print(f"  Fused kernel:   {'PASS' if passed_fused else 'FAIL'}")
    print(f"    Max abs error: {info_fused['max_abs_error']:.6e}")
    print(f"    Max rel error: {info_fused['max_rel_error']:.6e}")

    return passed and passed_fused


def benchmark_kernel(m, n, k, l=1, warmup=10, iters=100):
    """Benchmark custom kernel."""
    print(f"\n{'='*60}")
    print(f"Benchmarking M={m}, N={n}, K={k}, L={l}")
    print(f"{'='*60}")

    # Generate input
    data = generate_input(m, n, k, l, seed=42)

    # Warmup
    for _ in range(warmup):
        _ = custom_kernel_fused(data)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = custom_kernel_fused(data)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / iters
    elapsed_us = elapsed_ms * 1000

    # Compute FLOPS
    total_flops = 4 * m * n * k * l  # 2 GEMMs = 4*M*N*K
    tflops = (total_flops / (elapsed_ms / 1000)) / 1e12

    print(f"  Time: {elapsed_us:.3f} μs")
    print(f"  TFLOPS: {tflops:.2f}")

    return elapsed_us, tflops


if __name__ == "__main__":
    print("="*60)
    print("NVFP4 Dual-GEMM Kernel Validation")
    print("="*60)

    # Test cases matching SOL targets
    test_cases = [
        (256, 4096, 7168, 1),   # Target: 4.708 μs
        (512, 4096, 7168, 1),   # Target: 8.714 μs
        (256, 3072, 4096, 1),   # Target: 2.125 μs
        (512, 3072, 7168, 1),   # Target: 6.535 μs
    ]

    all_passed = True
    for m, n, k, l in test_cases:
        passed = test_kernel(m, n, k, l)
        all_passed = all_passed and passed

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("="*60)

    # Benchmark if tests pass
    if all_passed:
        print("\n\nBenchmarking:")
        sol_targets = {
            (256, 4096, 7168): 4.708,
            (512, 4096, 7168): 8.714,
            (256, 3072, 4096): 2.125,
            (512, 3072, 7168): 6.535,
        }

        for (m, n, k, l) in test_cases:
            elapsed_us, tflops = benchmark_kernel(m, n, k, l)
            target = sol_targets.get((m, n, k), None)
            if target:
                ratio = elapsed_us / target
                status = "✓" if elapsed_us < target * 1.1 else "✗"
                print(f"  vs SOL {target:.3f} μs: {ratio:.2f}x {status}")
