from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUTLASS include path
cutlass_include = os.path.join(os.path.dirname(__file__), '..', 'cutlass', 'include')

setup(
    name='nvfp4_dual_gemm_cuda',
    ext_modules=[
        CUDAExtension(
            'silu_mul_cuda',
            ['silu_mul_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_100a',
                    '--use_fast_math',
                    '-lineinfo',
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
