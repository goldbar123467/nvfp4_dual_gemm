# Comprehensive benchmark of submission.py

import torch
import sys
sys.path.insert(0, '/workspace/nvfp4_dual_gemm/python')
from task import generate_input, ref_kernel
from submission import custom_kernel

def benchmark_and_validate(m, n, k, l, seed=42, iters=100):
    """Benchmark and validate a single configuration."""
    data = generate_input(m, n, k, l, seed)

    # Validate correctness
    result = custom_kernel(data)
    ref = ref_kernel(data)
    match = torch.allclose(result, ref, rtol=1e-3, atol=1e-3)

    # Warmup
    for _ in range(10):
        _ = custom_kernel(data)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = custom_kernel(data)
    end.record()
    torch.cuda.synchronize()

    elapsed_us = (start.elapsed_time(end) / iters) * 1000

    return elapsed_us, match

# Problem sizes from task.md
configs = [
    (256, 4096, 7168, 1, 4.708),
    (512, 4096, 7168, 1, 8.714),
    (256, 3072, 4096, 1, 2.125),
    (512, 3072, 7168, 1, 6.535),
]

print("=" * 70)
print("NVFP4 Dual-GEMM Benchmark (submission.py)")
print("=" * 70)
print(f"{'Config':<25} {'Correct':>8} {'Time (μs)':>12} {'SOL (μs)':>12} {'Gap':>8}")
print("-" * 70)

times = []
for m, n, k, l, sol in configs:
    elapsed, correct = benchmark_and_validate(m, n, k, l)
    times.append(elapsed)
    status = "PASS" if correct else "FAIL"
    gap = f"{elapsed/sol:.1f}x"
    print(f"M={m:<4} N={n:<4} K={k:<4} L={l}   {status:>8}   {elapsed:>10.1f}   {sol:>10.3f}   {gap:>8}")

# Geometric mean
import math
geomean = math.exp(sum(math.log(t) for t in times) / len(times))
print("-" * 70)
print(f"{'Geometric Mean':<25} {'':>8}   {geomean:>10.1f}")
print("=" * 70)

# Leaderboard reference: 13.304 μs
print(f"\nLeaderboard top: 13.304 μs")
print(f"Our geomean: {geomean:.1f} μs")
print(f"Gap to top: {geomean/13.304:.1f}x")
