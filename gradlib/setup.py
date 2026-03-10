import os
import sys
import torch
from setuptools import setup

aiter_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{aiter_dir}/aiter/")
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    ROCM_HOME
)

this_dir = os.path.dirname(os.path.abspath(__file__))

# Detect whether this is a ROCm environment
is_rocm_pytorch = torch.version.hip is not None and ROCM_HOME is not None

# GPU architecture list (adjust as needed)
gpus = ["gfx90a", "gfx940", "gfx941", "gfx942"]
extra_args = ["--offload-arch=" + g for g in gpus]

# Version-dependent macros (optional, for compatibility with older PyTorch)
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])
version_ge_1_1 = [] if TORCH_MAJOR == 1 and TORCH_MINOR <= 0 else ["-DVERSION_GE_1_1"]
version_ge_1_3 = [] if TORCH_MAJOR == 1 and TORCH_MINOR <= 2 else ["-DVERSION_GE_1_3"]
version_ge_1_5 = [] if TORCH_MAJOR == 1 and TORCH_MINOR <= 4 else ["-DVERSION_GE_1_5"]
version_dependent_macros = version_ge_1_1 + version_ge_1_3 + version_ge_1_5

include_dirs = [os.path.join(this_dir, "csrc"),os.path.join(this_dir, "include"),]

ext_modules = []

# If in a ROCm environment, compile the hipBLASLt-related extensions
if is_rocm_pytorch:
    # Extension 1: hipbsolgemm based on hipBLASLt
    ext_modules.append(
        CUDAExtension(
            name="hipbsolidxgemm",  # keep original module name, modify if needed
            sources=["./csrc/hipbsolgemm.cu"],
            include_dirs=include_dirs,
            libraries=["hipblaslt"],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-DLEGACY_HIPBLAS_DIRECT=ON",
                    "-DENABLE_TORCH_FP8",
                ] + version_dependent_macros,
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-ftemplate-depth=1024",
                    "-DLEGACY_HIPBLAS_DIRECT=ON",
                    "-DENABLE_TORCH_FP8",
                ] + extra_args,
            },
        )
    )
    # Extension 2: rocsolgemm based on rocBLAS (optional, can be commented out if not needed)
    ext_modules.append(
        CUDAExtension(
            name="rocsolidxgemm",
            sources=["./csrc/rocsolgemm.cu"],
            include_dirs=include_dirs,
            libraries=["rocblas"],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-DLEGACY_HIPBLAS_DIRECT=ON",
                ] + version_dependent_macros,
                "nvcc": [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-ftemplate-depth=1024",
                    "-DLEGACY_HIPBLAS_DIRECT=ON",
                ] + extra_args,
            },
        )
    )
else:
    # Non-ROCm environment (CUDA) – not handled here; add code if needed
    pass

setup(
    name="gradlib",
    packages=["gradlib"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
