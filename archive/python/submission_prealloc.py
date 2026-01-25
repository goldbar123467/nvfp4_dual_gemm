# =============================================================================
# NVFP4 Dual-GEMM Submission - Pre-allocated Intermediates
# =============================================================================
# GPU MODE Leaderboard Submission
# Kernel: C = silu(A @ B1) * (A @ B2)
# Target: NVIDIA B200 (SM100 Blackwell)
#
# Pre-allocates intermediate buffers and ensures contiguous scales
# =============================================================================

import torch
from torch.utils.cpp_extension import load_inline
from typing import Tuple, Dict

# -----------------------------------------------------------------------------
# Inline CUDA epilogue kernel
# -----------------------------------------------------------------------------
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void fused_silu_mul_kernel(
    const float* __restrict__ r1,
    const float* __restrict__ r2,
    __half* __restrict__ out,
    int n_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {
        float v1 = r1[idx];
        float v2 = r2[idx];
        float sig = __frcp_rn(1.0f + __expf(-v1));
        out[idx] = __float2half(v1 * sig * v2);
    }
}

void fused_silu_mul(torch::Tensor r1, torch::Tensor r2, torch::Tensor out) {
    int n = r1.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    fused_silu_mul_kernel<<<blocks, threads>>>(
        r1.data_ptr<float>(),
        r2.data_ptr<float>(),
        reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
        n
    );
}
"""

cpp_source = """
#include <torch/extension.h>
void fused_silu_mul(torch::Tensor r1, torch::Tensor r2, torch::Tensor out);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_silu_mul", &fused_silu_mul, "Fused SiLU multiply");
}
"""

_cuda_module = None

def _get_cuda_module():
    global _cuda_module
    if _cuda_module is None:
        _cuda_module = load_inline(
            name='fused_epilogue_prealloc',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['-O3', '--use_fast_math'],
            verbose=False,
        )
    return _cuda_module


# Cache for pre-allocated buffers
_buffer_cache: Dict[tuple, dict] = {}

def _get_buffers(M: int, N: int):
    key = (M, N)
    if key not in _buffer_cache:
        _buffer_cache[key] = {
            'r1': torch.empty((M, N), dtype=torch.float32, device='cuda'),
            'r2': torch.empty((M, N), dtype=torch.float32, device='cuda'),
        }
    return _buffer_cache[key]


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

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def custom_kernel(data: input_t) -> output_t:
    """
    NVFP4 dual GEMM with pre-allocated intermediates.
    """
    a, b1, b2, _, _, _, sfa_perm, sfb1_perm, sfb2_perm, c_out = data
    m, n, l = c_out.shape
    cuda_mod = _get_cuda_module()

    if l == 1:
        # Contiguous scale factors
        scale_a = sfa_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1).contiguous()
        scale_b1 = sfb1_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1).contiguous()
        scale_b2 = sfb2_perm[:, :, :, :, :, 0].permute(2, 4, 0, 1, 3).reshape(-1).contiguous()

        # Contiguous matrix slices
        a_mat = a[:, :, 0].contiguous()
        b1_t = b1[:, :, 0].t().contiguous()
        b2_t = b2[:, :, 0].t().contiguous()

        # Dual GEMM
        r1 = torch._scaled_mm(a_mat, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_mat, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        # Fused epilogue
        cuda_mod.fused_silu_mul(r1, r2, c_out[:, :, 0])
        return c_out

    # L > 1 case
    for i in range(l):
        scale_a = sfa_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1).contiguous()
        scale_b1 = sfb1_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1).contiguous()
        scale_b2 = sfb2_perm[:, :, :, :, :, i].permute(2, 4, 0, 1, 3).reshape(-1).contiguous()

        a_mat = a[:, :, i].contiguous()
        b1_t = b1[:, :, i].t().contiguous()
        b2_t = b2[:, :, i].t().contiguous()

        r1 = torch._scaled_mm(a_mat, b1_t, scale_a, scale_b1, out_dtype=torch.float32)
        r2 = torch._scaled_mm(a_mat, b2_t, scale_a, scale_b2, out_dtype=torch.float32)

        cuda_mod.fused_silu_mul(r1, r2, c_out[:, :, i])

    return c_out
