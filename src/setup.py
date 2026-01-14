from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUTLASS include path
cutlass_include = os.path.join(os.path.dirname(__file__), '..', 'cutlass', 'include')
cutlass_tools = os.path.join(os.path.dirname(__file__), '..', 'cutlass', 'tools', 'util', 'include')

setup(
    name='nvfp4_dual_gemm_cuda',
    ext_modules=[
        CUDAExtension(
            'nvfp4_dual_gemm_cuda',
            [
                'nvfp4_dual_gemm.cpp',
                'kernel/nvfp4_dual_gemm.cu',
            ],
            include_dirs=[
                cutlass_include,
                cutlass_tools,
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '-arch=sm_100a',
                    '--expt-relaxed-constexpr',
                    '-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1',
                    '--use_fast_math',
                    '-lineinfo',
                    '-Xcompiler', '-fPIC',
                ]
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
