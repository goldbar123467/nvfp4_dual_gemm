// NVFP4 Dual GEMM with SiLU Fusion - PyTorch C++ Extension
// Computes: C = silu(A @ B1) * (A @ B2)
// Phase 1: C++ wrapper to eliminate Python dispatch overhead

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Forward declaration of fused SiLU*mul CUDA kernel
torch::Tensor silu_mul_fused_cuda(
    const torch::Tensor& input1,
    const torch::Tensor& input2
);

// Main dual GEMM function - uses torch._scaled_mm from C++
torch::Tensor nvfp4_dual_gemm_silu(
    const torch::Tensor& a,      // [M, K//2] FP4 packed
    const torch::Tensor& b1,     // [N, K//2] FP4 packed
    const torch::Tensor& b2,     // [N, K//2] FP4 packed
    const torch::Tensor& sfa,    // Scale factors for A (blocked format)
    const torch::Tensor& sfb1,   // Scale factors for B1 (blocked format)
    const torch::Tensor& sfb2    // Scale factors for B2 (blocked format)
) {
    // Ensure we're on CUDA
    TORCH_CHECK(a.is_cuda(), "Input a must be on CUDA");
    TORCH_CHECK(b1.is_cuda(), "Input b1 must be on CUDA");
    TORCH_CHECK(b2.is_cuda(), "Input b2 must be on CUDA");

    // Transpose B matrices for GEMM: [N, K] -> [K, N]
    auto b1_t = b1.t().contiguous();
    auto b2_t = b2.t().contiguous();

    // GEMM1: A @ B1.T using scaled_mm
    auto r1 = torch::_scaled_mm(
        a, b1_t, sfa, sfb1,
        /*bias=*/c10::nullopt,
        /*out_dtype=*/torch::kFloat32
    );

    // GEMM2: A @ B2.T using scaled_mm
    auto r2 = torch::_scaled_mm(
        a, b2_t, sfa, sfb2,
        /*bias=*/c10::nullopt,
        /*out_dtype=*/torch::kFloat32
    );

    // Fused SiLU + multiply using custom CUDA kernel
    return silu_mul_fused_cuda(r1, r2);
}

// Batched version for L > 1
torch::Tensor nvfp4_dual_gemm_silu_batched(
    const torch::Tensor& a,      // [M, K//2, L]
    const torch::Tensor& b1,     // [N, K//2, L]
    const torch::Tensor& b2,     // [N, K//2, L]
    const torch::Tensor& sfa,    // [M, K//16, L] or blocked
    const torch::Tensor& sfb1,   // [N, K//16, L] or blocked
    const torch::Tensor& sfb2    // [N, K//16, L] or blocked
) {
    const auto L = a.size(2);

    if (L == 1) {
        // Fast path for L=1
        auto result = nvfp4_dual_gemm_silu(
            a.squeeze(2), b1.squeeze(2), b2.squeeze(2),
            sfa.squeeze(2), sfb1.squeeze(2), sfb2.squeeze(2)
        );
        return result.unsqueeze(2);
    }

    // General case
    const auto M = a.size(0);
    const auto N = b1.size(0);
    auto output = torch::empty({M, N, L},
        torch::TensorOptions().dtype(torch::kFloat16).device(a.device()));

    for (int64_t l = 0; l < L; ++l) {
        auto result = nvfp4_dual_gemm_silu(
            a.select(2, l), b1.select(2, l), b2.select(2, l),
            sfa.select(2, l), sfb1.select(2, l), sfb2.select(2, l)
        );
        output.select(2, l).copy_(result);
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dual_gemm_silu", &nvfp4_dual_gemm_silu,
          "NVFP4 Dual GEMM with SiLU fusion (single slice)",
          py::arg("a"), py::arg("b1"), py::arg("b2"),
          py::arg("sfa"), py::arg("sfb1"), py::arg("sfb2"));

    m.def("dual_gemm_silu_batched", &nvfp4_dual_gemm_silu_batched,
          "NVFP4 Dual GEMM with SiLU fusion (batched)",
          py::arg("a"), py::arg("b1"), py::arg("b2"),
          py::arg("sfa"), py::arg("sfb1"), py::arg("sfb2"));
}
