# SHARK TANK ROUND 5: WILD CARD A
## The Triton Kernel Approach

---

## THE PITCH: Why Triton Could Win

### The Problem with CuTe DSL

The current CuTe kernel has fundamental issues:

1. **Launch Overhead Dominance**: At 400-530 microseconds, we are spending enormous time on kernel launch, setup, tensormap management, and pipeline coordination
2. **Two-Pass Architecture**: Running the kernel twice (GEMM1, then GEMM2) doubles all fixed costs
3. **Complex Abstraction Tax**: CuTe DSL generates verbose code with many indirections
4. **No Fusion**: SiLU activation runs as a separate PyTorch operation

### Why Triton Could Beat CuTe

| Factor | CuTe DSL | Triton |
|--------|----------|--------|
| Launch overhead | High (complex setup) | Low (JIT compiled) |
| Fusion potential | Requires kernel rewrite | Built-in tiling fusion |
| Development iteration | Slow (compile errors) | Fast (Python-like) |
| Hardware mapping | Manual tensormap/TMA | Automatic optimization |
| Dual GEMM fusion | Complex restructuring | Natural loop structure |

---

## TECHNICAL APPROACH

### Strategy: Fused Dual GEMM with Block Scaling in Triton

```python
@triton.jit
def nvfp4_dual_gemm_silu_kernel(
    # Pointers
    a_ptr, b1_ptr, b2_ptr,      # FP4 packed input tensors
    sfa_ptr, sfb1_ptr, sfb2_ptr, # FP8 scale factors
    c_ptr,                       # FP16 output
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak, stride_b1n, stride_b1k,
    stride_b2n, stride_b2k, stride_cm, stride_cn,
    # Scale factor strides
    stride_sfa_m, stride_sfa_k, stride_sfb_n, stride_sfb_k,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Computes: C = silu(A @ B1) * (A @ B2)

    NVFP4 format: 4-bit floats (e2m1), packed 2 per byte
    Scale factors: FP8 (e4m3fn), one per 16 elements
    """
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Swizzle for L2 cache efficiency
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers for A (shared between both GEMMs)
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulators in FP32
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k

        # Load A tile (FP4 packed -> unpack -> apply scale)
        # Note: This is where Triton's limitation hits - no native FP4 support
        a_packed = tl.load(a_ptr + offs_am[:, None] * stride_am +
                          (k_offs // 2)[None, :] * stride_ak)

        # Load scale factors for A (16 elements per scale)
        sfa_offs_m = offs_am // 128  # Scale factor layout matching
        sfa_offs_k = k_offs // 16
        sfa = tl.load(sfa_ptr + sfa_offs_m[:, None] * stride_sfa_m +
                      sfa_offs_k[None, :] * stride_sfa_k)

        # Unpack FP4 (this is the tricky part)
        # Low nibble = first element, high nibble = second element
        a_low = (a_packed & 0x0F).to(tl.float16)  # Simplified - need proper FP4 decode
        a_high = ((a_packed >> 4) & 0x0F).to(tl.float16)
        a = tl.interleave(a_low, a_high)  # Pseudocode

        # Apply scale factors
        a_scaled = a * sfa.to(tl.float16)

        # Load B1 tile with scales
        b1_packed = tl.load(b1_ptr + offs_bn[:, None] * stride_b1n +
                           (k_offs // 2)[None, :] * stride_b1k)
        sfb1 = tl.load(sfb1_ptr + ...)
        b1_scaled = unpack_and_scale_fp4(b1_packed, sfb1)

        # Load B2 tile with scales
        b2_packed = tl.load(b2_ptr + offs_bn[:, None] * stride_b2n +
                           (k_offs // 2)[None, :] * stride_b2k)
        sfb2 = tl.load(sfb2_ptr + ...)
        b2_scaled = unpack_and_scale_fp4(b2_packed, sfb2)

        # BOTH GEMMs in the same K-loop iteration (A tile reuse!)
        acc1 += tl.dot(a_scaled, b1_scaled.T)
        acc2 += tl.dot(a_scaled, b2_scaled.T)

    # Fused epilogue: C = silu(acc1) * acc2
    sigmoid_acc1 = tl.sigmoid(acc1)
    silu_acc1 = acc1 * sigmoid_acc1
    c = silu_acc1 * acc2

    # Store result in FP16
    c_fp16 = c.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c_fp16, mask=mask)
```

---

## THE CRITICAL CHALLENGE: FP4 Support in Triton

### The Elephant in the Room

Triton **does not natively support FP4 (float4_e2m1fn_x2)**. This is the single biggest risk.

### Potential Solutions

#### Option 1: Software FP4 Decode (REALISTIC)

```python
@triton.jit
def decode_fp4_e2m1(packed_byte):
    """
    Decode FP4 e2m1 format:
    - 1 sign bit
    - 2 exponent bits
    - 1 mantissa bit

    Values: [-1.5, -1.0, -0.5, 0, +0.5, +1.0, +1.5]
    """
    # FP4 e2m1 encoding (4 bits):
    # 0000 = +0.0
    # 0001 = +0.5
    # 0010 = +1.0
    # 0011 = +1.5
    # 1000 = -0.0
    # 1001 = -0.5
    # 1010 = -1.0
    # 1011 = -1.5

    # Lookup table approach (8 values)
    LUT = tl.constexpr([0.0, 0.5, 1.0, 1.5, 0.0, -0.5, -1.0, -1.5])

    low_nibble = packed_byte & 0x0F
    high_nibble = (packed_byte >> 4) & 0x0F

    val_low = LUT[low_nibble]
    val_high = LUT[high_nibble]

    return val_low, val_high
```

**Problem**: LUT approach may be slow. We need vectorized decode.

#### Option 2: Inline PTX for FP4->FP16 Conversion

```python
@triton.jit
def fp4_to_fp16_ptx(packed_fp4):
    """Use inline PTX if available for hardware FP4 conversion."""
    # Blackwell has hardware support - can we access it from Triton?
    # This would require Triton extension or inline assembly
    return tl.inline_asm(
        "cvt.rn.f16.e2m1 $0, $1;",
        "=h, h",
        [packed_fp4],
    )
```

**Problem**: Triton's inline PTX support is limited and version-dependent.

#### Option 3: Use Triton's Built-in Matmul with Pre-dequantized Data

```python
def preprocess_fp4_to_fp16(a_fp4, sfa):
    """Dequantize FP4 to FP16 before kernel (defeats purpose)."""
    # This works but loses the memory bandwidth benefit of FP4
    pass
```

**Problem**: This defeats the entire purpose of FP4.

---

## EXPECTED SPEEDUP ANALYSIS

### Best Case (If FP4 decode works efficiently)

| Component | CuTe Current | Triton Optimistic | Improvement |
|-----------|--------------|-------------------|-------------|
| Kernel launch | ~50 us | ~10 us | 5x |
| TMA/tensormap setup | ~100 us | 0 us (auto) | Eliminated |
| Two-pass overhead | ~100 us | 0 us (fused) | Eliminated |
| GEMM computation | ~150 us | ~150 us | Same |
| Epilogue (SiLU*mul) | ~30 us | 0 us (fused) | Eliminated |
| **Total** | **~430 us** | **~160 us** | **2.7x** |

**Optimistic target: ~160 us (still 8-10x from speed-of-light)**

### Realistic Case (With software FP4 decode overhead)

| Component | Estimate |
|-----------|----------|
| Kernel launch | ~10 us |
| GEMM with software FP4 decode | ~400-600 us |
| Fused epilogue | ~0 us |
| **Total** | **~410-610 us** |

**Realistic target: ~500 us (about same as current, possibly worse)**

### Why We Won't Hit 2-19 us

The speed-of-light targets assume:
- Perfect hardware utilization
- Zero overhead
- Optimal memory access patterns

Even with a perfect Triton kernel, we face:
1. **Software FP4 decode penalty**: ~3-5x slowdown vs hardware
2. **No TMA in Triton**: Must use regular loads/stores
3. **No TMEM access**: Can't use tensor memory
4. **Block size constraints**: 128x128 is still required

---

## RISKS AND CHALLENGES

### HIGH Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| No FP4 support | FATAL | Software decode or abandon |
| No MmaMXF4NVF4Op | FATAL | Cannot match CuTe performance |
| Block size mismatch | HIGH | May not tile efficiently |
| Scale factor layout | HIGH | Complex reordering needed |

### MEDIUM Risk Items

| Risk | Impact | Mitigation |
|------|--------|------------|
| Group GEMM batching | MEDIUM | Multiple kernel launches |
| Triton autotuning | MEDIUM | May not find optimal config |
| FP8 scale factor support | MEDIUM | May need custom handling |

### The Honest Assessment

**Triton is NOT the right tool for this job.**

The NVFP4 MMA instruction (`MmaMXF4NVF4Op`) is a specialized hardware unit that:
- Expects data in specific format (FP4 packed, K-major)
- Uses tensor memory (TMEM) for accumulators
- Requires 128x128 tile sizes
- Has hardware scale factor application

Triton cannot access any of these features. It would need to:
- Software decode every FP4 value (massive overhead)
- Use regular FP16/FP32 matmul (no FP4 tensor core)
- Miss out on TMA hardware acceleration

---

## ALTERNATIVE PROPOSAL: Triton for Epilogue Only

Instead of replacing the entire kernel, use Triton for just the fused epilogue:

```python
@triton.jit
def fused_silu_mul_kernel(
    gemm1_ptr, gemm2_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused epilogue: output = silu(gemm1) * gemm2

    This IS something Triton can do well!
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load GEMM results (FP32)
    g1 = tl.load(gemm1_ptr + offsets, mask=mask)
    g2 = tl.load(gemm2_ptr + offsets, mask=mask)

    # Fused SiLU * mul
    sigmoid_g1 = tl.sigmoid(g1)
    silu_g1 = g1 * sigmoid_g1
    result = silu_g1 * g2

    # Store as FP16
    tl.store(output_ptr + offsets, result.to(tl.float16), mask=mask)
```

### This Could Save ~30 us

The current implementation does:
```python
result = (torch.nn.functional.silu(temp1_fp32) * temp2_fp32).to(c.dtype)
```

A Triton fused kernel would:
- Eliminate intermediate tensor allocation
- Fuse 3 operations into 1 kernel
- Reduce memory traffic

**Expected savings: 10-30 us (small but real)**

---

## VERDICT: INVESTMENT RECOMMENDATION

### DO NOT INVEST in Full Triton Kernel

| Metric | Score |
|--------|-------|
| Technical Feasibility | 2/10 |
| Performance Potential | 3/10 |
| Implementation Risk | 9/10 |
| Time to MVP | 10+ hours |

**Reason**: Triton cannot access NVFP4 MMA hardware. Any Triton implementation would be fundamentally slower than CuTe.

### CONSIDER INVESTING in Triton Epilogue Fusion

| Metric | Score |
|--------|-------|
| Technical Feasibility | 9/10 |
| Performance Potential | 4/10 |
| Implementation Risk | 2/10 |
| Time to MVP | 1 hour |

**Reason**: This is a low-risk, incremental improvement that plays to Triton's strengths.

---

## CODE SKELETON: Triton Epilogue (What Could Actually Work)

```python
import triton
import triton.language as tl
import torch

@triton.jit
def _fused_silu_mul_kernel(
    gemm1_ptr, gemm2_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load in FP32 for numerical stability
    g1 = tl.load(gemm1_ptr + offsets, mask=mask).to(tl.float32)
    g2 = tl.load(gemm2_ptr + offsets, mask=mask).to(tl.float32)

    # Fused SiLU(g1) * g2
    sigmoid_g1 = tl.sigmoid(g1)
    result = (g1 * sigmoid_g1) * g2

    # Store as FP16
    tl.store(output_ptr + offsets, result.to(tl.float16), mask=mask)


def fused_silu_mul_triton(gemm1: torch.Tensor, gemm2: torch.Tensor) -> torch.Tensor:
    """Apply fused SiLU * mul using Triton."""
    assert gemm1.shape == gemm2.shape
    assert gemm1.is_contiguous() and gemm2.is_contiguous()

    output = torch.empty(gemm1.shape, dtype=torch.float16, device=gemm1.device)
    n_elements = gemm1.numel()

    # Autotune block size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _fused_silu_mul_kernel[grid](
        gemm1, gemm2, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


def solve_with_triton_epilogue(data):
    """
    Hybrid approach:
    - CuTe kernel for GEMMs (use hardware FP4)
    - Triton for fused epilogue
    """
    from submission import run_single_gemm

    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c = data
    m, n, l = c.shape
    k = a.shape[1] * 2
    problem_sizes = [(m, n, k, l)]

    # Allocate FP32 temps for maximum precision
    temp1 = torch.empty((m, n, l), dtype=torch.float32, device='cuda')
    temp2 = torch.empty((m, n, l), dtype=torch.float32, device='cuda')

    # CuTe GEMMs
    run_single_gemm(a, b1, sfa_perm, sfb1_perm, temp1, problem_sizes)
    run_single_gemm(a, b2, sfa_perm, sfb2_perm, temp2, problem_sizes)

    # Triton fused epilogue (the only part we can improve)
    result = fused_silu_mul_triton(temp1.flatten(), temp2.flatten())

    c.copy_(result.view(m, n, l))
    return c
```

---

## FINAL SUMMARY

### The Honest Truth

| Approach | Can it beat CuTe? | Why? |
|----------|-------------------|------|
| Full Triton GEMM | NO | No FP4 hardware access |
| Triton Epilogue Only | MARGINALLY | Saves ~10-30 us |
| CuTe Interleaved Dual GEMM | YES (potentially) | Reuses A tiles, proper hardware |

### My Recommendation as Wild Card A

**Don't bet on me for the full kernel.** I'm honest about my limitations.

If you want a Triton win, use me for the epilogue only. It's a small win, but it's a real win.

**The real performance gains will come from:**
1. Interleaved dual GEMM in CuTe (A-tile reuse)
2. Warp specialization
3. TMA store epilogue
4. Persistent kernel / Stream-K

These all require staying within the CuTe DSL ecosystem to access NVFP4 MMA hardware.

---

*"Know your limitations. The best tool for FP4 is the hardware designed for FP4."*
*- Wild Card A, accepting defeat gracefully*
