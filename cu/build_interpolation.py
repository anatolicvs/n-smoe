from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="cpp_interpolation",
    version="0.0.1",
    author="Aytac Ozkan",
    author_email="aytac@linux.com",
    description="A cpp extension for custom interpolation operations in PyTorch",
    ext_modules=[
        CUDAExtension(
            name="cpp_interpolation",
            sources=["interpolation.cpp"],
            extra_compile_args={
                "cxx": ["-O2", "-Wall"],
                "nvcc": ["-O2", "-arch=sm_75"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch>=1.8.0",
        "numpy",
    ],
)
