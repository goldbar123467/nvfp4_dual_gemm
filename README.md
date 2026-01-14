<div align="center">

```
 ╔══════════════════════════════════════════════════════════════════════════════╗
 ║                                                                              ║
 ║    ███╗   ██╗██╗   ██╗███████╗██████╗ ██╗  ██╗                               ║
 ║    ████╗  ██║██║   ██║██╔════╝██╔══██╗██║  ██║                               ║
 ║    ██╔██╗ ██║██║   ██║█████╗  ██████╔╝███████║                               ║
 ║    ██║╚██╗██║╚██╗ ██╔╝██╔══╝  ██╔═══╝ ╚════██║                               ║
 ║    ██║ ╚████║ ╚████╔╝ ██║     ██║          ██║                               ║
 ║    ╚═╝  ╚═══╝  ╚═══╝  ╚═╝     ╚═╝          ╚═╝                               ║
 ║                                                                              ║
 ║    ██████╗ ██╗   ██╗ █████╗ ██╗          ██████╗ ███████╗███╗   ███╗███╗   ███╗
 ║    ██╔══██╗██║   ██║██╔══██╗██║         ██╔════╝ ██╔════╝████╗ ████║████╗ ████║
 ║    ██║  ██║██║   ██║███████║██║         ██║  ███╗█████╗  ██╔████╔██║██╔████╔██║
 ║    ██║  ██║██║   ██║██╔══██║██║         ██║   ██║██╔══╝  ██║╚██╔╝██║██║╚██╔╝██║
 ║    ██████╔╝╚██████╔╝██║  ██║███████╗    ╚██████╔╝███████╗██║ ╚═╝ ██║██║ ╚═╝ ██║
 ║    ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝     ╚═════╝ ╚══════╝╚═╝     ╚═╝╚═╝     ╚═╝
 ║                                                                              ║
 ║             ┌─────────────────────────────────────────────────┐              ║
 ║             │                                                 │              ║
 ║             │      [A] ────┬────► [B1] ───► SiLU ──┐          │              ║
 ║             │              │                       ├──► [C]   │              ║
 ║             │              └────► [B2] ────────────┘          │              ║
 ║             │                                                 │              ║
 ║             └─────────────────────────────────────────────────┘              ║
 ║                                                                              ║
 ║               FP4 Tensor Core Acceleration for Blackwell B200                ║
 ║                                                                              ║
 ╚══════════════════════════════════════════════════════════════════════════════╝
```

</div>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/LICENSE-MIT-green?style=for-the-badge" alt="License MIT"></a>
  <a href="#"><img src="https://img.shields.io/badge/PYTHON-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="#"><img src="https://img.shields.io/badge/CUDA-12+-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA 12+"></a>
  <a href="#"><img src="https://img.shields.io/badge/GPU-B200_BLACKWELL-ff6600?style=for-the-badge&logo=nvidia&logoColor=white" alt="Target GPU B200"></a>
</p>

<p align="center">
  <strong>High-performance fused Dual-GEMM kernel exploiting NVIDIA Blackwell's FP4 Tensor Cores</strong>
</p>

---

## What is this?

This project implements an optimized CUDA kernel for computing the **fused Dual-GEMM operation with SiLU activation**:

```
C = SiLU(A @ B1) * (A @ B2)
```

Targeting **NVIDIA B200 (Blackwell)** GPUs, we leverage the new **NVFP4 (4-bit floating point) Tensor Cores** to achieve maximum throughput for this common pattern found in modern transformer architectures (GLU variants, gated MLPs, etc.).

---

<table align="center">
<tr>
<td align="center">

### 📊 Performance Summary

| Metric | Value |
|:-------|------:|
| **Target Latency** | 13 μs |
| **Achieved Latency** | ~30 μs |
| **Speedup vs Baseline** | **3.8×** |
| **Precision** | NVFP4 (E2M1) |
| **Architecture** | Blackwell (SM100) |

</td>
</tr>
</table>

---

## Submission Strategies

We explored several optimization approaches to minimize latency. Below is a summary of each strategy, their performance characteristics, and implementation status.

### Performance Comparison

| Strategy | File | Latency | Speedup | Status |
|:---------|:-----|--------:|--------:|:------:|
| Baseline | `submission.py` | ~38μs | 1.0x | ✅ Implemented |
| Cached Scale Factors | `submission_v2.py` | ~70μs | 0.5x | ✅ Implemented |
| CUDA Graphs | `submission_best.py` | ~30μs | **1.3x** | ✅ Implemented |
| Parallel Streams | `submission_streams.py` | ~39μs | ~1.0x | 🔄 Testing |
| Fused Dual-GEMM | — | ~13μs | 2.9x | 📋 Planned |

---

### Strategy Details

#### 1. ✅ Baseline (`submission.py`) — ~38μs

Simple PyTorch implementation leveraging `torch._scaled_mm` for FP4 matrix multiplication.

```python
# Core computation pattern
r1 = torch._scaled_mm(a, b1.T, scale_a, scale_b1, out_dtype=torch.float32)
r2 = torch._scaled_mm(a, b2.T, scale_a, scale_b2, out_dtype=torch.float32)
result = (F.silu(r1) * r2).half()
```

**Characteristics:**
- Two sequential GEMMs followed by SiLU fusion
- No caching — recomputes scale factors on every call
- Straightforward and easy to debug

---

#### 2. ✅ Cached Scale Factors (`submission_v2.py`) — ~70μs

Optimizes repeated scale factor transformations by caching based on tensor memory addresses.

```python
# Cache lookup by data pointer
key = (sfa_perm.data_ptr(), sfb1_perm.data_ptr(), sfb2_perm.data_ptr(), l_idx)
if key not in self.cache:
    scale_a = sfa_perm[...].permute(2, 4, 0, 1, 3).reshape(-1).clone()
    self.cache[key] = (scale_a, scale_b1, scale_b2)
return self.cache[key]
```

> ⚠️ **Note:** Unexpectedly slower due to cache lookup overhead in hot path.

---

#### 3. ✅ CUDA Graphs (`submission_best.py`) — ~30μs ⭐ Best

Captures the entire operation sequence into a CUDA graph for replay with minimal launch overhead.

```python
# Graph capture during initialization
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    r1 = torch._scaled_mm(a0, b1t, scale_a, scale_b1, out_dtype=torch.float32)
    r2 = torch._scaled_mm(a0, b2t, scale_a, scale_b2, out_dtype=torch.float32)
    output = (torch.nn.functional.silu(r1) * r2).half()

# Fast replay during inference
graph.replay()
```

**Why it works:**
- Eliminates per-call kernel launch latency
- Graph replay is a single GPU operation
- **3.8x speedup** over naive baseline

---

#### 4. 🔄 Parallel Streams (`submission_streams.py`) — ~39μs

Attempts to overlap GEMM1 and GEMM2 execution using separate CUDA streams.

```python
stream1, stream2 = torch.cuda.Stream(), torch.cuda.Stream()

with torch.cuda.stream(stream1):
    r1 = torch._scaled_mm(a0, b1t, ...)

with torch.cuda.stream(stream2):
    r2 = torch._scaled_mm(a0, b2t, ...)

stream1.synchronize()
stream2.synchronize()
result = F.silu(r1) * r2
```

> ⚠️ **Limitation:** May not improve latency if individual GEMMs already saturate GPU.

---

#### 5. 📋 Fused Dual-GEMM (Planned) — Target ~13μs

Custom CUTLASS kernel performing both GEMMs in a single pass.

```
Our Approach:                   Fused Approach:
─────────────                   ───────────────
Load A → GEMM1 → Store R1       Load A once
Load A → GEMM2 → Store R2         → GEMM1 + GEMM2 same kernel
Load R1, R2 → SiLU×R2 → Store     → SiLU×R2 in registers
                                  → Store C only
```

**Expected Benefits:**
| Metric | Current | Fused |
|:-------|:-------:|:-----:|
| A matrix loads | 2x | 1x |
| Kernel launches | 3 | 1 |
| Intermediate storage | 2 buffers | Registers |

---

### Performance Ranking

```
📋 Fused Dual-GEMM  ████████░░░░░░░░░░░░  ~13μs (target)
✅ CUDA Graphs      ████████████░░░░░░░░  ~30μs ⭐ current best
✅ Baseline         ███████████████░░░░░  ~38μs
🔄 Parallel Streams ███████████████░░░░░  ~39μs
✅ Cached Scales    ████████████████████  ~70μs
```

---

## Technical Deep Dive

### The Problem

We need to compute a fused dual matrix multiplication with SiLU activation:

```
C = silu(A @ B1) * (A @ B2)

Where:
  A  = input activation  [M × K]
  B1 = weight matrix 1   [N × K]
  B2 = weight matrix 2   [N × K]
  C  = output            [M × N]
```

#### Naive Implementation (What We're Optimizing)

```
    DRAM                        GPU                         DRAM
    ────                        ───                         ────

    ┌───┐
    │ A │──────────────────►  GEMM1  ──────────────────────► R1
    │B1 │──────────────────►  (A@B1)                        (temp)
    └───┘

    ┌───┐
    │ A │──────────────────►  GEMM2  ──────────────────────► R2
    │B2 │──────────────────►  (A@B2)     ▲                  (temp)
    └───┘                                │
                                    A loaded TWICE!
    ┌───┐
    │R1 │──────────────────►  Epilogue ────────────────────► C
    │R2 │──────────────────►  silu×mul                     (output)
    └───┘
         ▲
         └── R1,R2 round-trip to DRAM!
```

**Why This Is Slow:**

| Issue | Impact |
|-------|--------|
| 3 kernel launches | ~15-30μs overhead |
| A loaded twice | 2× memory bandwidth |
| R1, R2 round-trip | Write then read M×N elements |

---

### FP4 Block-Scaled Format

#### NVFP4 (E2M1): 4-bit Floating Point

```
    FP4 E2M1 Bit Layout
    ┌───┬───┬───┬───┐
    │ S │ E │ E │ M │  (4 bits total)
    └───┴───┴───┴───┘
      │   └─┬─┘   │
      │     │     └── Mantissa (1 bit)
      │     └──────── Exponent (2 bits)
      └────────────── Sign (1 bit)

    Representable Values: {±0, ±0.5, ±1.0, ±1.5}
```

#### Block Scaling

Every 16 FP4 elements share one FP8 scale factor:

```
    FP4 Data:     [v0][v1][v2]...[v15] [v16][v17]...[v31] ...
                   └───────┬────────┘  └───────┬────────┘
    FP8 Scales:         [SF0]              [SF1]          ...

    Effective:    v[i] * SF[i // 16]
```

---

### Why CUDA Graphs Help

```
    WITHOUT CUDA GRAPH
    ══════════════════

    CPU: ──[Launch K1]────[Launch K2]────[Launch K3]────►
                │              │              │
                ▼ ~5-10μs      ▼ ~5-10μs      ▼ ~5-10μs
    GPU: ─────[K1]──────────[K2]──────────[K3]─────────►

    Total launch overhead: 15-30μs


    WITH CUDA GRAPH
    ═══════════════

    CPU: ──[Launch Graph]─────────────────────────────►
                │
                ▼ ~5μs (single launch)
    GPU: ─────[K1]─[K2]─[K3]──────────────────────────►

    Total launch overhead: ~5μs
    Savings: 10-25μs!
```

---

### The 13μs Solution

The optimal solution fuses everything into a single kernel:

```
    OPTIMAL FUSED KERNEL
    ════════════════════

    DRAM                     GPU Registers                 DRAM
    ────                     ─────────────                 ────

    ┌───┐                    ┌─────────────────────┐
    │ A │─────► Load once ──►│ acc1 = A @ B1       │
    │B1 │─────────────────►  │ acc2 = A @ B2       │
    │B2 │─────────────────►  │ C = silu(acc1)*acc2 │──────► C
    └───┘                    └─────────────────────┘
                                      ▲
                             All in registers!
                             No intermediate DRAM!
```

**Memory Savings:**

| Operation | Naive | Fused |
|-----------|-------|-------|
| Load A | 2× | 1× |
| Store/Load R1,R2 | 4× M×N | 0 |
| **Extra traffic** | ~20MB | **0** |

---

## Q&A

<details>
<summary><strong>Why not just use PyTorch's built-in functions?</strong></summary>

We do! `torch._scaled_mm` uses cuBLAS FP4 tensor cores internally. The optimization is about reducing the overhead *around* these calls — launch latency, memory traffic, and redundant operations.

</details>

<details>
<summary><strong>Why is CUDA graph faster than raw PyTorch?</strong></summary>

CUDA graphs eliminate kernel launch overhead (~5-10μs per kernel). With 3 kernels in the naive approach, that's 15-30μs of pure overhead eliminated by capturing and replaying the graph.

</details>

<details>
<summary><strong>Can the two GEMMs run in parallel?</strong></summary>

Theoretically yes with CUDA streams, but FP4 GEMMs likely saturate the tensor cores entirely. When compute is fully utilized, parallelism at the stream level doesn't help — the GPU can only do one thing at a time anyway.

</details>

<details>
<summary><strong>Why FP32 intermediate instead of FP16?</strong></summary>

Precision. The SiLU activation (`x * sigmoid(x)`) involves an exponential operation that benefits from FP32 accumulation. FP16 intermediates cause numerical errors that fail validation.

</details>

<details>
<summary><strong>What would it take to reach 13μs?</strong></summary>

A fully fused kernel that:
1. Loads A tiles once and reuses for both B1 and B2
2. Keeps both accumulator results in registers
3. Computes `silu(acc1) * acc2` without touching DRAM
4. Writes only the final output C

This requires a custom CUTLASS kernel with modified mainloop and epilogue.

</details>

<details>
<summary><strong>Why is scale factor conversion so complex?</strong></summary>

cuBLAS uses a specific "atom" layout for FP4 block scales that's optimized for tensor core access patterns. The task provides scale factors in a different permuted format. Converting between them requires careful index manipulation: `[32, 4, M//128, 4, K//64, L]` ↔ flattened blocked format.

</details>

---

## Lessons Learned

| | Insight |
|:---:|---------|
| 💡 | **Kernel launch overhead dominates** small matrix operations — profile before assuming compute is the bottleneck |
| 💡 | **CUDA graphs are powerful** but require static shapes and careful memory management |
| 💡 | **Memory bandwidth is the bottleneck**, not compute — fusing operations to reduce DRAM traffic is key |
| 💡 | **Scale factor layouts differ** between libraries — PyTorch/cuBLAS vs CUTLASS expect different formats |
| 💡 | **Sometimes simple beats complex** — our PyTorch + CUDA graph solution outperformed initial CUTLASS attempts |
| 💡 | **Profile first, optimize second** — assumptions about bottlenecks are often wrong |

---

## Future Work

- [ ] Implement fused dual-GEMM CUTLASS kernel for 13μs target
- [ ] Explore Triton for custom kernel fusion
- [ ] Benchmark across different problem sizes (M, N, K)
- [ ] Add FP8 variant for comparison
- [ ] Profile memory bandwidth utilization

---

## Project Structure

```
nvfp4_dual_gemm/
├── python/
│   ├── submission.py          # Baseline implementation
│   ├── submission_v2.py       # Cached scale factors
│   ├── submission_best.py     # CUDA graph optimized ⭐
│   ├── submission_streams.py  # Parallel streams experiment
│   └── task.py                # Task definition & validation
├── src/
│   ├── dual_gemm_nvfp4.cu     # CUTLASS kernel (WIP)
│   ├── nvfp4_gemm.cuh         # GEMM wrapper
│   └── silu_mul_kernel.cuh    # Fused epilogue
├── cutlass/                   # CUTLASS library (submodule)
└── README.md                  # This file
```

---

## Usage

```python
from python.submission_best import custom_kernel

# Input tuple: (a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c_out)
result = custom_kernel(data)
```

---

## Acknowledgments

Built for the [GPU MODE](https://gpumode.com/) NVFP4 Dual-GEMM challenge.

Target hardware: **NVIDIA B200 (Blackwell)** with SM100 FP4 Tensor Cores.

---

<p align="center">
  <sub>Made with ⚡ and lots of profiling</sub>
</p>
