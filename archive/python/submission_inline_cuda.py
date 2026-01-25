# =============================================================================
# NVFP4 Dual-GEMM Submission - Inline CUDA Version
# =============================================================================
# GPU MODE Leaderboard Submission
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
# =============================================================================

import torch
from torch.utils.cpp_extension import load_inline
from typing import Tuple

# -----------------------------------------------------------------------------
# Inline CUDA source for fused epilogue
# -----------------------------------------------------------------------------
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Fused kernel: out[i] = silu(r1[i]) * r2[i] = r1[i] * sigmoid(r1[i]) * r2[i]
// Reads float32, writes float16
__global__ void fused_silu_mul_kernel(
    const float* __restrict__ r1,
    const float* __restrict__ r2,
    __half* __restrict__ out,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Process 4 elements per thread for better memory coalescing
    int base_idx = idx * 4;

    if (base_idx + 3 < n_elements) {
        // Load 4 elements from each input
        float4 v1 = *reinterpret_cast<const float4*>(r1 + base_idx);
        float4 v2 = *reinterpret_cast<const float4*>(r2 + base_idx);

        // Compute silu(v1) * v2 for each element
        __half results[4];

        float sig0 = 1.0f / (1.0f + expf(-v1.x));
        float sig1 = 1.0f / (1.0f + expf(-v1.y));
        float sig2 = 1.0f / (1.0f + expf(-v1.z));
        float sig3 = 1.0f / (1.0f + expf(-v1.w));

        results[0] = __float2half(v1.x * sig0 * v2.x);
        results[1] = __float2half(v1.y * sig1 * v2.y);
        results[2] = __float2half(v1.z * sig2 * v2.z);
        results[3] = __float2half(v1.w * sig3 * v2.w);

        // Store as half2 for coalesced writes
        *reinterpret_cast<__half2*>(out + base_idx) = make_half2(results[0], results[1]);
        *reinterpret_cast<__half2*>(out + base_idx + 2) = make_half2(results[2], results[3]);
    } else {
        // Handle boundary
        for (int i = 0; i < 4 && base_idx + i < n_elements; i++) {
            float v1_val = r1[base_idx + i];
            float v2_val = r2[base_idx + i];
            float sig = 1.0f / (1.0f + expf(-v1_val));
            out[base_idx + i] = __float2half(v1_val * sig * v2_val);
        }
    }
}

// Faster version using __expf intrinsic
__global__ void fused_silu_mul_kernel_fast(
    const float* __restrict__ r1,
    const float* __restrict__ r2,
    __half* __restrict__ out,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        float v1 = r1[idx];
        float v2 = r2[idx];
        float sig = __frcp_rn(1.0f + __expf(-v1));  // Fast reciprocal and exp
        out[idx] = __float2half(v1 * sig * v2);
    }
}

torch::Tensor fused_silu_mul_cuda(torch::Tensor r1, torch::Tensor r2) {
    auto n = r1.numel();
    auto out = torch::empty({r1.size(0), r1.size(1)},
                           torch::dtype(torch::kFloat16).device(r1.device()));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    fused_silu_mul_kernel_fast<<<blocks, threads>>>(
        r1.data_ptr<float>(),
        r2.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        n
    );

    return out;
}

// In-place version that writes to pre-allocated output
void fused_silu_mul_cuda_inplace(
    torch::Tensor r1,
    torch::Tensor r2,
    torch::Tensor out
) {
    auto n = r1.numel();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    fused_silu_mul_kernel_fast<<<blocks, threads>>>(
        r1.data_ptr<float>(),
        r2.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        n
    );
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor fused_silu_mul_cuda(torch::Tensor r1, torch::Tensor r2);
void fused_silu_mul_cuda_inplace(torch::Tensor r1, torch::Tensor r2, torch::Tensor out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_silu_mul", &fused_silu_mul_cuda, "Fused SiLU multiply (CUDA)");
    m.def("fused_silu_mul_inplace", &fused_silu_mul_cuda_inplace, "Fused SiLU multiply inplace (CUDA)");
}
"""

# Compile the inline CUDA module
_cuda_module = None

def get_cuda_module():
    global _cuda_module
    if _cuda_module is None:
        _cuda_module = load_inline(
            name='fused_epilogue',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return _cuda_module


# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
input_t = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]
output_t = torch.Tensor


def custom_kernel(data: input_t) -> output_t:
    """
    Optimized NVFP4 block-scaled dual GEMM with SiLU activation.

    Uses inline CUDA for fused epilogue kernel.
    """
    a, b1, b2, _, _, _, sfa_perm, sfb1_perm, sfb2_perm, c_out = data

    m, n, l = c_out.shape
    cuda_mod = get_cuda_module()

    if l == 1:
        # Convert scale factors
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)

        # Matrix slices
        a_mat = a[:, :, 0]
        b1_t = b1[:, :, 0].t()
        b2_t = b2[:, :, 0].t()

        # Dual GEMM using cuBLAS FP4 tensor cores
        r1 = torch._scaled_mm(a_mat, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_mat, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        # Fused CUDA epilogue - writes directly to c_out
        cuda_mod.fused_silu_mul_inplace(r1, r2, c_out[:, :, 0])

        return c_out

    # L > 1 case
    for l_idx in range(l):
        scale_a = sfa_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, l_idx].permute(2, 4, 0, 1, 3).reshape(-1)

        a_mat = a[:, :, l_idx]
        b1_t = b1[:, :, l_idx].t()
        b2_t = b2[:, :, l_idx].t()

        r1 = torch._scaled_mm(a_mat, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_mat, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        cuda_mod.fused_silu_mul_inplace(r1, r2, c_out[:, :, l_idx])

    return c_out
