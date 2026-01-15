# =============================================================================
# NVFP4 Dual-GEMM Submission - V6 Vectorized Epilogue
# =============================================================================
import torch
from torch.utils.cpp_extension import load_inline
from typing import Tuple

# Vectorized epilogue - processes 4 elements per thread
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

// Scalar fallback for non-aligned sizes
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
            name='fused_epilogue_v6',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return _cuda_module


input_t = Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]
output_t = torch.Tensor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, _, _, _, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape
    cuda_mod = _get_cuda_module()

    if l == 1:
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1)

        a_mat = a[:, :, 0]
        b1_t = b1[:, :, 0].t()
        b2_t = b2[:, :, 0].t()

        r1 = torch._scaled_mm(a_mat, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_mat, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        cuda_mod.fused_silu_mul(r1, r2, c_out[:, :, 0])
        return c_out

    for i in range(l):
        scale_a = sfa_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b1 = sfb1_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1)
        scale_b2 = sfb2_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1)

        a_mat = a[:, :, i]
        b1_t = b1[:, :, i].t()
        b2_t = b2[:, :, i].t()

        r1 = torch._scaled_mm(a_mat, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_mat, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        cuda_mod.fused_silu_mul(r1, r2, c_out[:, :, i])

    return c_out
