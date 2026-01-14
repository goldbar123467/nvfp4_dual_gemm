// Fused SiLU and Multiply CUDA Kernel
// Computes: output = silu(input1) * input2
// Where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Fast sigmoid approximation
__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Fused SiLU * Multiply kernel for FP32 inputs, FP16 output
template <typename scalar_t>
__global__ void silu_mul_kernel(
    half* __restrict__ output,
    const scalar_t* __restrict__ input1,
    const scalar_t* __restrict__ input2,
    const int64_t size
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x1 = static_cast<float>(input1[idx]);
        float x2 = static_cast<float>(input2[idx]);

        // SiLU(x1) * x2 = (x1 * sigmoid(x1)) * x2
        float sigmoid_x1 = fast_sigmoid(x1);
        float result = x1 * sigmoid_x1 * x2;

        output[idx] = __float2half(result);
    }
}

// Vectorized version for better memory throughput
__global__ void silu_mul_kernel_vec4(
    float4* __restrict__ output,
    const float4* __restrict__ input1,
    const float4* __restrict__ input2,
    const int64_t size
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float4 x1 = input1[idx];
        float4 x2 = input2[idx];

        float4 result;
        result.x = x1.x * fast_sigmoid(x1.x) * x2.x;
        result.y = x1.y * fast_sigmoid(x1.y) * x2.y;
        result.z = x1.z * fast_sigmoid(x1.z) * x2.z;
        result.w = x1.w * fast_sigmoid(x1.w) * x2.w;

        output[idx] = result;
    }
}

// Main launcher function
torch::Tensor silu_mul_fused(
    const torch::Tensor& input1,  // [M, N] FP32 - result of GEMM1
    const torch::Tensor& input2   // [M, N] FP32 - result of GEMM2
) {
    TORCH_CHECK(input1.is_cuda(), "input1 must be on CUDA");
    TORCH_CHECK(input2.is_cuda(), "input2 must be on CUDA");
    TORCH_CHECK(input1.sizes() == input2.sizes(), "inputs must have same shape");

    auto output = torch::empty(input1.sizes(),
        torch::TensorOptions().dtype(torch::kFloat16).device(input1.device()));

    const int64_t size = input1.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input1.scalar_type(), "silu_mul_kernel", ([&] {
        silu_mul_kernel<scalar_t><<<blocks, threads>>>(
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            input1.data_ptr<scalar_t>(),
            input2.data_ptr<scalar_t>(),
            size
        );
    }));

    return output;
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_mul", &silu_mul_fused, "Fused SiLU and multiply (CUDA)");
}
