#!/usr/bin/env python3
"""Test script for NVFP4 Dual GEMM CUDA kernel."""

import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../python')

import torch
import time

def test_correctness():
    """Test kernel correctness against reference implementation."""
    print("=" * 60)
    print("Testing NVFP4 Dual GEMM CUDA Kernel")
    print("=" * 60)

    try:
        import nvfp4_dual_gemm_cuda
        print("[OK] Module imported successfully")
    except ImportError as e:
        print(f"[FAIL] Failed to import module: {e}")
        print("\nTry building with: pip install -e .")
        return False

    from task import generate_input, ref_kernel

    # Test problem size
    M, N, K, L = 512, 4096, 7168, 1
    print(f"\nTest size: M={M}, N={N}, K={K}, L={L}")

    # Generate input
    data = generate_input(M, N, K, L, seed=42)
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c = data

    # Get scale factors in blocked format (from permuted)
    # Convert: [32, 4, rest_m, 4, rest_k, L] -> blocked format
    scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
    scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
    scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)

    # Run our kernel
    print("\nRunning CUDA kernel...")
    result = nvfp4_dual_gemm_cuda.dual_gemm_silu(
        a[:, :, 0], b1[:, :, 0], b2[:, :, 0],
        scale_a, scale_b1, scale_b2
    )

    # Run reference
    print("Running reference kernel...")
    ref = ref_kernel(data)

    # Compare
    ref_slice = ref[:, :, 0]
    correct = torch.allclose(result, ref_slice, rtol=1e-3, atol=1e-3)
    max_diff = (result - ref_slice).abs().max().item()
    mean_diff = (result - ref_slice).abs().mean().item()

    print(f"\n{'='*40}")
    print(f"Result shape: {result.shape}")
    print(f"Reference shape: {ref_slice.shape}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Correct (rtol=1e-3, atol=1e-3): {correct}")
    print(f"{'='*40}")

    return correct


def benchmark():
    """Benchmark kernel performance."""
    print("\n" + "=" * 60)
    print("Benchmarking")
    print("=" * 60)

    import nvfp4_dual_gemm_cuda
    from task import generate_input

    # Test sizes from leaderboard
    test_sizes = [
        (256, 4096, 7168, 1),
        (512, 4096, 7168, 1),
        (256, 3072, 4096, 1),
        (512, 3072, 7168, 1),
    ]

    for M, N, K, L in test_sizes:
        data = generate_input(M, N, K, L, seed=42)
        a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c = data

        # Get blocked scale factors
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)

        a_s = a[:, :, 0]
        b1_s = b1[:, :, 0]
        b2_s = b2[:, :, 0]

        # Warmup
        for _ in range(10):
            nvfp4_dual_gemm_cuda.dual_gemm_silu(a_s, b1_s, b2_s, scale_a, scale_b1, scale_b2)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            nvfp4_dual_gemm_cuda.dual_gemm_silu(a_s, b1_s, b2_s, scale_a, scale_b1, scale_b2)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 100 * 1e6

        print(f"M={M}, N={N}, K={K}: {elapsed:.1f} μs")

    print("\nTarget: <20 μs | Leaderboard #1: 13.3 μs")


if __name__ == "__main__":
    if test_correctness():
        print("\n[PASS] Correctness test passed!")
        benchmark()
    else:
        print("\n[FAIL] Correctness test failed!")
        sys.exit(1)
