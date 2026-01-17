# =============================================================================
# NVFP4 Dual-GEMM Submission - V7 CUTLASS Integration (B200/SM100)
# =============================================================================
"""
Target: NVIDIA B200 (SM100 - Datacenter Blackwell)

Computes: C = silu(A @ B1.T) * (A @ B2.T)

Key changes from V6:
- Primary: cutlass_nvfp4_mm from gn-kernels (optimal NVFP4 performance)
- Fallback: torch._scaled_mm with proper to_blocked() scale conversion
- Same vectorized fused SiLU*mul epilogue kernel
"""

import torch
from torch.utils.cpp_extension import load_inline
from typing import Tuple

# =============================================================================
# Try to import gn-kernels CUTLASS
# =============================================================================
HAS_GN_KERNELS = False

try:
    from gn_kernels import cutlass_nvfp4_mm
    HAS_GN_KERNELS = True
except ImportError:
    pass

# Also try direct SM100 import if available
if not HAS_GN_KERNELS:
    try:
        from gn_kernels._sm100a import cutlass_nvfp4_mm
        HAS_GN_KERNELS = True
    except ImportError:
        pass


# =============================================================================
# Scale factor utilities (from reference implementation)
# =============================================================================
def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix):
    """
    Convert scale factor tensor to blocked format for torch._scaled_mm.

    This matches the cuBLAS block scaling factors layout:
    https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
    """
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    blocks = input_matrix.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


# =============================================================================
# Vectorized Epilogue Kernel - fused silu(r1) * r2
# =============================================================================
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_silu_mul_vec4(
    const float4* __restrict__ r1,
    const float4* __restrict__ r2,
    __half2* __restrict__ out,
    int n_vec4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec4) {
        float4 v1 = r1[idx];
        float4 v2 = r2[idx];

        // silu(x) * y = x * sigmoid(x) * y
        float s0 = __frcp_rn(1.0f + __expf(-v1.x));
        float s1 = __frcp_rn(1.0f + __expf(-v1.y));
        float s2 = __frcp_rn(1.0f + __expf(-v1.z));
        float s3 = __frcp_rn(1.0f + __expf(-v1.w));

        __half h0 = __float2half(v1.x * s0 * v2.x);
        __half h1 = __float2half(v1.y * s1 * v2.y);
        __half h2 = __float2half(v1.z * s2 * v2.z);
        __half h3 = __float2half(v1.w * s3 * v2.w);

        out[idx * 2] = __halves2half2(h0, h1);
        out[idx * 2 + 1] = __halves2half2(h2, h3);
    }
}

__global__ void fused_silu_mul_scalar(
    const float* __restrict__ r1,
    const float* __restrict__ r2,
    __half* __restrict__ out,
    int start,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < n_elements) {
        float v1 = r1[idx];
        float v2 = r2[idx];
        float sig = __frcp_rn(1.0f + __expf(-v1));
        out[idx] = __float2half(v1 * sig * v2);
    }
}

void fused_silu_mul(torch::Tensor r1, torch::Tensor r2, torch::Tensor out) {
    int n = r1.numel();
    int n_vec4 = n / 4;
    int remainder = n % 4;

    if (n_vec4 > 0) {
        int threads = 256;
        int blocks = (n_vec4 + threads - 1) / threads;
        fused_silu_mul_vec4<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(r1.data_ptr<float>()),
            reinterpret_cast<const float4*>(r2.data_ptr<float>()),
            reinterpret_cast<__half2*>(out.data_ptr<at::Half>()),
            n_vec4
        );
    }

    if (remainder > 0) {
        int start = n_vec4 * 4;
        int threads = 32;
        int blocks = 1;
        fused_silu_mul_scalar<<<blocks, threads>>>(
            r1.data_ptr<float>(),
            r2.data_ptr<float>(),
            reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
            start, n
        );
    }
}
"""

cpp_source = """
#include <torch/extension.h>
void fused_silu_mul(torch::Tensor r1, torch::Tensor r2, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_silu_mul", &fused_silu_mul, "Fused SiLU multiply vectorized");
}
"""

_cuda_module = None


def _get_cuda_module():
    global _cuda_module
    if _cuda_module is None:
        _cuda_module = load_inline(
            name='fused_epilogue_v7_sm100',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return _cuda_module


# =============================================================================
# Type definitions
# =============================================================================
input_t = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,  # a, b1, b2 (float4_e2m1fn_x2)
    torch.Tensor, torch.Tensor, torch.Tensor,  # sfa_ref, sfb1_ref, sfb2_ref (float8_e4m3fn)
    torch.Tensor, torch.Tensor, torch.Tensor,  # sfa_perm, sfb1_perm, sfb2_perm (permuted)
    torch.Tensor,  # c_out (float16)
]
output_t = torch.Tensor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# =============================================================================
# Main kernel function
# =============================================================================
def custom_kernel(data: input_t) -> output_t:
    """
    NVFP4 Dual-GEMM with fused SiLU*mul epilogue.

    Computes: C = silu(A @ B1.T) * (A @ B2.T)

    Uses CUTLASS NVFP4 GEMM if gn-kernels available, else torch._scaled_mm.
    """
    # Unpack inputs
    # a, b1, b2: [M/N, K//2, L] in float4_e2m1fn_x2
    # sfa_ref, sfb1_ref, sfb2_ref: [M/N, K//16, L] in float8_e4m3fn (unpermuted)
    # sfa_perm, sfb1_perm, sfb2_perm: permuted layout for CUTLASS
    # c_out: [M, N, L] output buffer
    a, b1, b2, sfa_ref, sfb1_ref, sfb2_ref, sfa_perm, sfb1_perm, sfb2_perm, c_out = data

    m, n, l = c_out.shape
    cuda_mod = _get_cuda_module()

    if HAS_GN_KERNELS:
        # =================================================================
        # CUTLASS Path (Optimal for B200)
        # =================================================================
        return _cutlass_kernel(
            a, b1, b2, sfa_ref, sfb1_ref, sfb2_ref, c_out, m, n, l, cuda_mod
        )
    else:
        # =================================================================
        # Fallback Path (torch._scaled_mm with to_blocked)
        # =================================================================
        return _fallback_kernel(
            a, b1, b2, sfa_ref, sfb1_ref, sfb2_ref, c_out, m, n, l, cuda_mod
        )


def _cutlass_kernel(a, b1, b2, sfa_ref, sfb1_ref, sfb2_ref, c_out, m, n, l, cuda_mod):
    """
    CUTLASS NVFP4 GEMM path.

    cutlass_nvfp4_mm expects:
    - A: [M, K//2] contiguous
    - B: [N, K//2] with B.T contiguous
    - scale_A: [M, K//16] float8_e4m3fn
    - scale_B: [N, K//16] float8_e4m3fn
    - output_scale: scalar tensor
    """
    for i in range(l):
        # Extract batch slice
        a_mat = a[:, :, i].contiguous()         # [M, K//2]
        b1_mat = b1[:, :, i].contiguous()       # [N, K//2]
        b2_mat = b2[:, :, i].contiguous()       # [N, K//2]

        # Scale factors - use unpermuted [M/N, K//16] format for CUTLASS
        scale_a = sfa_ref[:, :, i].contiguous()    # [M, K//16]
        scale_b1 = sfb1_ref[:, :, i].contiguous()  # [N, K//16]
        scale_b2 = sfb2_ref[:, :, i].contiguous()  # [N, K//16]

        # Ensure float8_e4m3fn dtype
        if scale_a.dtype != torch.float8_e4m3fn:
            scale_a = scale_a.to(torch.float8_e4m3fn)
            scale_b1 = scale_b1.to(torch.float8_e4m3fn)
            scale_b2 = scale_b2.to(torch.float8_e4m3fn)

        # Output scale (no additional scaling)
        output_scale = torch.tensor(1.0, device=a_mat.device)

        try:
            # CUTLASS NVFP4 GEMMs: (M, K//2) @ (N, K//2).T -> (M, N)
            r1 = cutlass_nvfp4_mm(a_mat, b1_mat.t(), scale_a, scale_b1, output_scale)
            r2 = cutlass_nvfp4_mm(a_mat, b2_mat.t(), scale_a, scale_b2, output_scale)

            # Convert to float32 for epilogue if needed
            if r1.dtype != torch.float32:
                r1 = r1.float()
                r2 = r2.float()

            # Fused SiLU * mul epilogue
            cuda_mod.fused_silu_mul(r1, r2, c_out[:, :, i])

        except Exception as e:
            # Fall back to torch._scaled_mm on failure
            print(f"CUTLASS failed: {e}, using fallback")
            _fallback_single_batch(
                a[:, :, i], b1[:, :, i], b2[:, :, i],
                sfa_ref[:, :, i], sfb1_ref[:, :, i], sfb2_ref[:, :, i],
                c_out[:, :, i], cuda_mod
            )

    return c_out


def _fallback_kernel(a, b1, b2, sfa_ref, sfb1_ref, sfb2_ref, c_out, m, n, l, cuda_mod):
    """
    Fallback using torch._scaled_mm with proper to_blocked() scale conversion.
    """
    for i in range(l):
        _fallback_single_batch(
            a[:, :, i], b1[:, :, i], b2[:, :, i],
            sfa_ref[:, :, i], sfb1_ref[:, :, i], sfb2_ref[:, :, i],
            c_out[:, :, i], cuda_mod
        )

    return c_out


def _fallback_single_batch(a_mat, b1_mat, b2_mat, sfa, sfb1, sfb2, c_out_slice, cuda_mod):
    """
    Single batch fallback using torch._scaled_mm.

    IMPORTANT: torch._scaled_mm requires scales in blocked format via to_blocked()
    """
    # Convert scales to blocked format (this was missing in V6!)
    # to_blocked expects CPU tensors, so move to CPU first if needed
    sfa_cpu = sfa.cpu() if sfa.is_cuda else sfa
    sfb1_cpu = sfb1.cpu() if sfb1.is_cuda else sfb1
    sfb2_cpu = sfb2.cpu() if sfb2.is_cuda else sfb2

    scale_a = to_blocked(sfa_cpu).cuda()
    scale_b1 = to_blocked(sfb1_cpu).cuda()
    scale_b2 = to_blocked(sfb2_cpu).cuda()

    # GEMM: (M, K//2) @ (N, K//2).T -> (M, N)
    r1 = torch._scaled_mm(
        a_mat,
        b1_mat.t(),
        scale_a,
        scale_b1,
        bias=None,
        out_dtype=torch.float32,
    )

    r2 = torch._scaled_mm(
        a_mat,
        b2_mat.t(),
        scale_a,
        scale_b2,
        bias=None,
        out_dtype=torch.float32,
    )

    # Fused SiLU * mul epilogue
    cuda_mod.fused_silu_mul(r1, r2, c_out_slice)
