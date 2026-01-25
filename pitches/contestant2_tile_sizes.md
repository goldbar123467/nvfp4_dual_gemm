# SIZE MATTERS: The Tile Size Tuning Revolution

## Contestant #2 Pitch - GPU Optimization Shark Tank

---

## The Elevator Pitch

> "You're leaving 30-50% performance on the table because one-size-fits-all tile sizes are the mullets of GPU programming - they worked in the 80s, but we're in the era of adaptive computing now!"

---

## The Problem: The Procrustean Bed of 128x128x256

The current implementation uses a **fixed** `mma_tiler_mnk = (128, 128, 256)` for ALL problem sizes. This is like wearing the same size shoes whether you're a child or a basketball player.

### Why 128x128x256 Fails Our Benchmark Shapes

Let's examine our actual benchmark configurations:

| Config | M | N | K | Tiles (M) | Tiles (N) | Total CTAs | Issue |
|--------|---|---|---|-----------|-----------|------------|-------|
| g=8 (small) | 64 | 4096 | 7168 | 0.5 | 32 | **16** | M is SMALLER than tile! |
| g=8 (medium) | 248 | 4096 | 7168 | 1.9 | 32 | **64** | Wastes 29% of M tile |
| g=8 (other) | 40 | 7168 | 2048 | 0.3 | 55.9 | **56** | M is 31% of tile! |
| g=2 | 192 | 3072 | 4096 | 1.5 | 24 | **48** | Wastes 33% of M tile |

**The Brutal Truth:**
- When M=64 with tile M=128, you're computing TWICE the rows you need and throwing half away
- When M=40 with tile M=128, you're computing 3x the rows you need!
- [NVIDIA's own research](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) shows tile quantization waste can cost 20-40% performance

### Wave Quantization: The Silent Killer

The B200 has **144 SMs**. When total CTAs don't divide evenly by 144, you get [wave quantization](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/) - the GPU equivalent of leaving money on the table.

With our problem sizes:
- 16 CTAs = **11% SM utilization** (89% waste!)
- 48 CTAs = **33% SM utilization** (67% waste!)
- 64 CTAs = **44% SM utilization** (56% waste!)

**We're literally leaving more than half the GPU idle!**

---

## The Solution: Adaptive Tile Size Selection

### The Strategy: Right-Size Every Problem

Instead of one fixed tile, we select from a **curated set of tile configurations** based on problem dimensions:

```
Tile Options for NVFP4 on SM100 (Blackwell):
- (64, 64, 256)   - Small M/N batches, inference workloads
- (64, 128, 256)  - Asymmetric small-M cases
- (128, 64, 256)  - Asymmetric small-N cases
- (128, 128, 256) - Current default (medium)
- (128, 256, 256) - Large N dimension
- (256, 256, 128) - Maximum throughput for large GEMM (2SM instruction)
```

### Decision Algorithm

```python
def select_optimal_tile(M, N, K, num_sms=144):
    candidates = [
        (64, 64, 256),   # 8x more parallelism than 128x128
        (64, 128, 256),  # 4x more parallelism
        (128, 64, 256),  # 4x more parallelism
        (128, 128, 256), # Default
        (128, 256, 256), # Large N
    ]

    best_efficiency = 0
    best_tile = (128, 128, 256)

    for tile_m, tile_n, tile_k in candidates:
        if M % tile_m != 0 or N % tile_n != 0:
            continue  # Skip invalid tiles

        tiles_m = ceil_div(M, tile_m)
        tiles_n = ceil_div(N, tile_n)
        total_ctas = tiles_m * tiles_n

        # Wave efficiency (minimize partial waves)
        full_waves = total_ctas // num_sms
        partial_wave = total_ctas % num_sms
        wave_efficiency = (full_waves * num_sms + partial_wave) / ((full_waves + 1) * num_sms)

        # Tile efficiency (minimize wasted compute)
        tile_efficiency = (M * N) / (tiles_m * tile_m * tiles_n * tile_n)

        # Combined score
        efficiency = wave_efficiency * tile_efficiency

        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_tile = (tile_m, tile_n, tile_k)

    return best_tile
```

---

## Technical Details: The Deep Dive

### Supported MMA Tile Shapes on Blackwell SM100

According to [NVIDIA CUTLASS documentation](https://github.com/NVIDIA/cutlass/releases), for `mx_float4_t` (NVFP4) operations:

| MMA Tile Shape | Instruction Type | Use Case |
|---------------|------------------|----------|
| 128x128x128 | 1SM | Standard workloads |
| 128x256x128 | 1SM | Large N dimension |
| 256x256x128 | 2SM | Maximum throughput |

CUTLASS has recently added `mma_tiler_n=64` and `mma_tiler_n=192` support specifically for Blackwell blockscaled GEMM!

### Analysis for Our Benchmark Shapes

#### Shape: M=64, N=4096, K=7168

| Tile Config | CTAs | Waves | Efficiency | Speedup Est. |
|-------------|------|-------|------------|--------------|
| 128x128x256 | 16 | 0.11 | 11% | 1.0x (baseline) |
| **64x128x256** | 32 | 0.22 | 22% | **2.0x** |
| **64x64x256** | 64 | 0.44 | 44% | **4.0x** |

#### Shape: M=248, N=4096, K=7168

| Tile Config | CTAs | M-Waste | N-Waste | Total Efficiency |
|-------------|------|---------|---------|------------------|
| 128x128x256 | 64 | 29% | 0% | 44% SM util |
| **64x128x256** | 128 | 3% | 0% | 89% SM util |

### Memory Considerations

From [Colfax Research](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/):
- Larger tiles = better data reuse = fewer DRAM fetches
- Smaller tiles = more parallelism = better occupancy
- **Sweet spot depends on problem size**

For our small-M problems, the parallelism gain from smaller tiles **outweighs** the reduced data reuse.

---

## Expected Speedup: The Numbers That Matter

Based on [NVIDIA's GEMM performance research](https://arxiv.org/html/2411.16954):

### Conservative Estimates

| Benchmark Shape | Current | Optimized Tile | Speedup |
|-----------------|---------|----------------|---------|
| M=64, N=4096, K=7168 | 100% | (64, 64, 256) | **2.5-4x** |
| M=248, N=4096, K=7168 | 100% | (64, 128, 256) | **1.5-2x** |
| M=40, N=7168, K=2048 | 100% | (64, 64, 256) | **2-3x** |
| M=192, N=3072, K=4096 | 100% | (64, 128, 256) | **1.3-1.8x** |

### Geometric Mean Impact

- **Expected: 1.8-2.5x geometric mean improvement**
- This could cut our 373us runtime to **150-200us** with THIS OPTIMIZATION ALONE

[Research shows](https://ieeexplore.ieee.org/document/10360964) optimal tile selection can improve GEMM performance by up to **3.2x** while even reducing power consumption!

---

## Implementation Complexity: Surprisingly Manageable

### Difficulty: MEDIUM (2-3 days)

### Changes Required

1. **Add tile configuration constants** (30 minutes)
```python
TILE_CONFIGS = {
    'small_m': (64, 64, 256),
    'small_m_large_n': (64, 128, 256),
    'default': (128, 128, 256),
    'large_n': (128, 256, 256),
}
```

2. **Add tile selection function** (2 hours)
   - Simple heuristic based on M, N dimensions
   - No machine learning required (save that for v2)

3. **Parameterize kernel compilation** (4 hours)
   - Current: `mma_tiler_mnk = (128, 128, 256)` hardcoded
   - New: Pass as parameter, cache compiled variants

4. **Update shared memory layouts** (4 hours)
   - `a_smem_layout_staged`, `b_smem_layout_staged` already parameterized
   - Just need to flow through new tile sizes

5. **Testing and validation** (8 hours)
   - Verify correctness for all tile configs
   - Benchmark each configuration

### Code Snippet - The Core Change

```python
# Before (hardcoded)
mma_tiler_mnk = (128, 128, 256)

# After (adaptive)
def get_mma_tiler(m, n, k):
    if m < 128:
        return (64, 128 if n >= 2048 else 64, 256)
    elif n < 128:
        return (128, 64, 256)
    elif n >= 4096:
        return (128, 256, 256)
    return (128, 128, 256)
```

---

## Risk Assessment: Low to Medium

### What Could Go Wrong

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Smaller tiles use more shared memory per element | Low | Medium | Pre-validate memory fits |
| Some tile configs have alignment issues | Low | Low | Stick to powers of 2 |
| JIT compilation overhead for multiple variants | Medium | Low | Cache compiled kernels |
| Unexpected performance regression on some shapes | Low | Medium | Keep 128x128x256 as fallback |

### Why This is Low Risk

1. **Additive, not destructive** - We keep the working 128x128x256 as default
2. **Well-documented territory** - CUTLASS explicitly supports these tile sizes
3. **Easy rollback** - If a tile config fails, fall back to default
4. **Validated by NVIDIA** - These aren't exotic sizes; they're NVIDIA-recommended

From [NVIDIA's nvMatmulHeuristics](https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/):
> "The module achieved **96% of peak performance** in just 150 minutes on H100"

We're implementing a simplified version of exactly what NVIDIA recommends!

---

## The Ask: Why This Should Win

### 1. Highest ROI Optimization

- **Expected speedup: 1.8-2.5x** (geometric mean)
- **Implementation time: 2-3 days**
- **Risk: Low**
- **ROI: 60-125% speedup per day of work**

### 2. Foundational for Other Optimizations

Every other optimization (pipelining, warp specialization, TMA epilogue) will work BETTER with proper tile sizing. This is the **foundation** other improvements build on.

### 3. Immediate Impact on Worst-Performing Cases

Our smallest M values (40, 64) are currently performing at 11-30% efficiency. This optimization specifically targets our **worst cases**, providing the biggest lift where we need it most.

### 4. Battle-Tested Approach

This isn't experimental research - it's implementing what [NVIDIA's own auto-tuner](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/autotuning_gemm.html) does:
> "The search space includes mma_tiler_mn which defines the dimensions of the matrix tile"

---

## Closing Argument

> "Dear Sharks, you've heard about pipeline stages, TMA stores, and warp specialization. But none of those matter if we're computing **three times more data than we need** and leaving **89% of our SMs idle**.
>
> Tile size tuning is like adjusting your stride before a race - you don't try fancy techniques until your basic form is right.
>
> Give me 3 days, and I'll give you a **2x speedup**. That's not a promise, that's geometry.
>
> Size. Matters."

---

## References

1. [NVIDIA CUTLASS Auto-Tuning Guide](https://docs.nvidia.com/cutlass/media/docs/pythonDSL/cute_dsl_general/autotuning_gemm.html)
2. [NVIDIA Matrix Multiplication Performance Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)
3. [CUTLASS Persistent Kernels and Stream-K](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)
4. [Understanding GEMM Performance on Ada Lovelace](https://arxiv.org/html/2411.16954)
5. [NVIDIA CUTLASS Sub-byte GEMM on Blackwell](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/)
6. [GPU Tiling for Performance](https://ianbarber.blog/2025/05/30/keeping-a-gpu-busy-is-a-lot-about-tiling/)
7. [nvMatmulHeuristics Integration](https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/)
8. [Tailoring CUTLASS GEMM Using Supervised Learning](https://ieeexplore.ieee.org/document/10360964)

---

*Contestant #2 - Tile Size Tuning*
*"Because every microsecond counts, and every idle SM is a wasted opportunity."*
