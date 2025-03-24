#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup, find_packages
from torch import cuda
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

packages = ['instantsplat'] + ["instantsplat." + package for package in find_packages(where="instantsplat")]
packages_dust3r = ['dust3r'] + ["dust3r." + package for package in find_packages(where="submodules/dust3r/dust3r")]
packages_mast3r = ['mast3r'] + ["mast3r." + package for package in find_packages(where="submodules/mast3r/mast3r")]
packages_croco = ['croco', 'croco.utils', 'croco.models', 'croco.models.curope']

packages_dust3r += ["dust3r.dust3r"]  # ugly workaround for agly MAST3R import
os.makedirs("submodules/dust3r/dust3r/dust3r", exist_ok=True)  # ugly workaround for ugly MAST3R import
with open("submodules/dust3r/dust3r/dust3r/__init__.py", "w") as f:
    f.write("\n")

cxx_compiler_flags = []
nvcc_compiler_flags = []

# compile for all possible CUDA architectures
all_cuda_archs = cuda.get_gencode_flags().replace('compute=', 'arch=').split()
# alternatively, you can list cuda archs that you want, eg:
# all_cuda_archs = [
#     '-gencode', 'arch=compute_70,code=sm_70',
#     '-gencode', 'arch=compute_75,code=sm_75',
#     '-gencode', 'arch=compute_80,code=sm_80',
#     '-gencode', 'arch=compute_86,code=sm_86'
# ]

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")
    nvcc_compiler_flags.append("-allow-unsupported-compiler")

setup(
    name="instantsplat",
    version='1.2.0',
    author='yindaheng98',
    author_email='yindaheng98@gmail.com',
    url='https://github.com/yindaheng98/instantsplat',
    description=u'Refactored python initialization and training code for InstantSplat',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=packages + packages_dust3r + packages_mast3r + packages_croco,
    package_dir={
        'instantsplat': 'instantsplat',
        'dust3r': 'submodules/dust3r/dust3r',
        'mast3r': 'submodules/mast3r/mast3r',
        'croco': 'submodules/dust3r/croco',
    },
    ext_modules=[
        CUDAExtension(
            name='croco.models.curope.curope',
            sources=[
                "submodules/dust3r/croco/models/curope/curope.cpp",
                "submodules/dust3r/croco/models/curope/kernels.cu",
            ],
            extra_compile_args=dict(
                nvcc=nvcc_compiler_flags+['-O3', '--ptxas-options=-v', "--use_fast_math"]+all_cuda_archs,
                cxx=['-O3'])
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'opencv-python',
        'roma',
        'scipy',
        'huggingface_hub',
        'einops',
        'open3d',
        'gaussian-splatting',
    ]
)

os.remove("submodules/dust3r/dust3r/dust3r/__init__.py")  # ugly workaround for ugly MAST3R import
