// Fused SiLU and Multiply CUDA Kernel
// Computes: output = silu(input1) * input2
// Where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Vectorized SiLU + Multiply kernel
// Processes 4 elements per thread for better memory throughput
__global__ void silu_mul_kernel_vec4(
    half* __restrict__ output,
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    const int64_t size
) {
    const int64_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < size) {
        // Load 4 elements from each input
        float4 x1 = *reinterpret_cast<const float4*>(input1 + idx);
        float4 x2 = *reinterpret_cast<const float4*>(input2 + idx);

        // Compute silu(x1) * x2 for each element
        // silu(x) = x / (1 + exp(-x))
        float r0 = x1.x / (1.0f + expf(-x1.x)) * x2.x;
        float r1 = x1.y / (1.0f + expf(-x1.y)) * x2.y;
        float r2 = x1.z / (1.0f + expf(-x1.z)) * x2.z;
        float r3 = x1.w / (1.0f + expf(-x1.w)) * x2.w;

        // Convert to half and store
        half2 out01 = __floats2half2_rn(r0, r1);
        half2 out23 = __floats2half2_rn(r2, r3);

        *reinterpret_cast<half2*>(output + idx) = out01;
        *reinterpret_cast<half2*>(output + idx + 2) = out23;
    }
    else if (idx < size) {
        // Handle tail elements
        for (int i = 0; i < 4 && idx + i < size; i++) {
            float x1 = input1[idx + i];
            float x2 = input2[idx + i];
            float silu_x1 = x1 / (1.0f + expf(-x1));
            output[idx + i] = __float2half(silu_x1 * x2);
        }
    }
}

// Scalar fallback kernel for non-aligned sizes
__global__ void silu_mul_kernel_scalar(
    half* __restrict__ output,
    const float* __restrict__ input1,
    const float* __restrict__ input2,
    const int64_t size
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < size) {
        float x1 = input1[idx];
        float x2 = input2[idx];

        // SiLU(x1) * x2 = (x1 * sigmoid(x1)) * x2
        float sigmoid_x1 = 1.0f / (1.0f + expf(-x1));
        float result = x1 * sigmoid_x1 * x2;

        output[idx] = __float2half(result);
    }
}

// Main launcher function - called from C++
torch::Tensor silu_mul_fused_cuda(
    const torch::Tensor& input1,  // [M, N] FP32 - result of GEMM1
    const torch::Tensor& input2   // [M, N] FP32 - result of GEMM2
) {
    TORCH_CHECK(input1.is_cuda(), "input1 must be on CUDA");
    TORCH_CHECK(input2.is_cuda(), "input2 must be on CUDA");
    TORCH_CHECK(input1.sizes() == input2.sizes(), "inputs must have same shape");
    TORCH_CHECK(input1.scalar_type() == torch::kFloat32, "input1 must be FP32");
    TORCH_CHECK(input2.scalar_type() == torch::kFloat32, "input2 must be FP32");

    auto output = torch::empty(input1.sizes(),
        torch::TensorOptions().dtype(torch::kFloat16).device(input1.device()));

    const int64_t size = input1.numel();
    const int threads = 256;

    // Use vectorized kernel when size is divisible by 4
    if (size % 4 == 0) {
        const int blocks = (size / 4 + threads - 1) / threads;
        silu_mul_kernel_vec4<<<blocks, threads>>>(
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            input1.data_ptr<float>(),
            input2.data_ptr<float>(),
            size
        );
    } else {
        const int blocks = (size + threads - 1) / threads;
        silu_mul_kernel_scalar<<<blocks, threads>>>(
            reinterpret_cast<half*>(output.data_ptr<at::Half>()),
            input1.data_ptr<float>(),
            input2.data_ptr<float>(),
            size
        );
    }

    return output;
}
