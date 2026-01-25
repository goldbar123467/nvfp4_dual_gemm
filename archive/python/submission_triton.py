# =============================================================================
# NVFP4 Dual-GEMM Submission - Triton/Optimized Version
# =============================================================================
# Attempts multiple optimization strategies:
# 1. CUDA Graphs - capture and replay to eliminate launch overhead
# 2. Triton kernel - fused dual GEMM with shared A loading
# =============================================================================

import torch
from typing import Tuple

# Try to import triton
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Type aliases
input_t = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]
output_t = torch.Tensor

# Global cache for CUDA graphs
_graph_cache = {}


def _get_scale_factors(sfa_perm, sfb1_perm, sfb2_perm, l_idx=0):
    """Convert permuted scale factors to blocked format."""
    scale_a = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)
    scale_b1 = sfb1_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)
    scale_b2 = sfb2_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)
    return scale_a, scale_b1, scale_b2


def custom_kernel_cuda_graph(data: input_t) -> output_t:
    """
    CUDA Graph optimized version.
    Captures the dual GEMM + SiLU + mul sequence and replays it.
    """
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape

    if l != 1:
        # Fall back to loop for L > 1
        return _fallback_kernel(data)

    # Get key for cache (based on tensor shapes)
    cache_key = (m, n, a.shape[1] * 2)  # K = packed_k * 2

    if cache_key not in _graph_cache:
        # First call - need to capture the graph
        # Get scale factors
        scale_a, scale_b1, scale_b2 = _get_scale_factors(sfa_perm, sfb1_perm, sfb2_perm)

        a_slice = a[:, :, 0]
        b1_t = b1[:, :, 0].T.contiguous()
        b2_t = b2[:, :, 0].T.contiguous()

        # Warmup runs (required before graph capture)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
                r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=torch.float32)
                out = (torch.nn.functional.silu(r1) * r2).to(torch.float16)
        torch.cuda.current_stream().wait_stream(s)

        # Allocate static tensors for graph
        static_a = a_slice.clone()
        static_b1_t = b1_t.clone()
        static_b2_t = b2_t.clone()
        static_scale_a = scale_a.clone()
        static_scale_b1 = scale_b1.clone()
        static_scale_b2 = scale_b2.clone()
        static_output = torch.empty((m, n), dtype=torch.float16, device='cuda')

        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            r1 = torch._scaled_mm(static_a, static_b1_t, static_scale_a, static_scale_b1, out_dtype=torch.float32)
            r2 = torch._scaled_mm(static_a, static_b2_t, static_scale_a, static_scale_b2, out_dtype=torch.float32)
            static_output.copy_((torch.nn.functional.silu(r1) * r2).to(torch.float16))

        _graph_cache[cache_key] = {
            'graph': g,
            'static_a': static_a,
            'static_b1_t': static_b1_t,
            'static_b2_t': static_b2_t,
            'static_scale_a': static_scale_a,
            'static_scale_b1': static_scale_b1,
            'static_scale_b2': static_scale_b2,
            'static_output': static_output,
        }

    # Get cached graph and tensors
    cache = _graph_cache[cache_key]

    # Copy input data to static tensors
    scale_a, scale_b1, scale_b2 = _get_scale_factors(sfa_perm, sfb1_perm, sfb2_perm)

    cache['static_a'].copy_(a[:, :, 0])
    cache['static_b1_t'].copy_(b1[:, :, 0].T)
    cache['static_b2_t'].copy_(b2[:, :, 0].T)
    cache['static_scale_a'].copy_(scale_a)
    cache['static_scale_b1'].copy_(scale_b1)
    cache['static_scale_b2'].copy_(scale_b2)

    # Replay graph
    cache['graph'].replay()

    return cache['static_output'].unsqueeze(-1)


def _fallback_kernel(data: input_t) -> output_t:
    """Fallback for L > 1 or when optimizations fail."""
    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape

    output = torch.empty((m, n, l), dtype=torch.float16, device="cuda")

    for l_idx in range(l):
        scale_a, scale_b1, scale_b2 = _get_scale_factors(sfa_perm, sfb1_perm, sfb2_perm, l_idx)

        a_slice = a[:, :, l_idx]
        b1_t = b1[:, :, l_idx].T
        b2_t = b2[:, :, l_idx].T

        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        output[:, :, l_idx] = (torch.nn.functional.silu(r1) * r2).to(torch.float16)

    return output


# =============================================================================
# Triton Fused SiLU * Mul Kernel (for epilogue optimization)
# =============================================================================

if HAS_TRITON:
    @triton.jit
    def _silu_mul_kernel(
        output_ptr,
        input1_ptr,
        input2_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused SiLU(input1) * input2 kernel."""
        pid = tl.program_id(0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load inputs
        x1 = tl.load(input1_ptr + offsets, mask=mask)
        x2 = tl.load(input2_ptr + offsets, mask=mask)

        # SiLU(x1) = x1 * sigmoid(x1)
        sigmoid_x1 = tl.sigmoid(x1)
        silu_x1 = x1 * sigmoid_x1

        # Output = SiLU(x1) * x2
        output = silu_x1 * x2

        # Store as FP16
        tl.store(output_ptr + offsets, output.to(tl.float16), mask=mask)

    def silu_mul_triton(input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """Apply fused SiLU * mul using Triton."""
        assert input1.shape == input2.shape
        output = torch.empty(input1.shape, dtype=torch.float16, device=input1.device)
        n_elements = input1.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        _silu_mul_kernel[grid](
            output, input1, input2, n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return output


def custom_kernel_triton_epilogue(data: input_t) -> output_t:
    """
    Uses torch._scaled_mm for GEMMs but Triton for fused epilogue.
    """
    if not HAS_TRITON:
        return _fallback_kernel(data)

    a, b1, b2, sfa, sfb1, sfb2, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape

    if l == 1:
        scale_a, scale_b1, scale_b2 = _get_scale_factors(sfa_perm, sfb1_perm, sfb2_perm)

        a_slice = a[:, :, 0]
        b1_t = b1[:, :, 0].T
        b2_t = b2[:, :, 0].T

        # Dual GEMM
        r1 = torch._scaled_mm(a_slice, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_slice, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        # Fused epilogue with Triton
        output = silu_mul_triton(r1, r2)
        return output.unsqueeze(-1)

    return _fallback_kernel(data)


# =============================================================================
# Main entry point - try optimizations in order
# =============================================================================

def custom_kernel(data: input_t) -> output_t:
    """
    Main entry point. Tries optimizations in order:
    1. CUDA Graphs (if supported)
    2. Triton epilogue
    3. Fallback to pure PyTorch
    """
    # Try Triton epilogue (simpler, more reliable than full CUDA graphs)
    if HAS_TRITON:
        return custom_kernel_triton_epilogue(data)

    # Fallback
    return _fallback_kernel(data)
