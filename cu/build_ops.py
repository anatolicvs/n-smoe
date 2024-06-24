from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_block_ops',
    version='0.0.1',
    author='Aytac Ozkan',
    author_email='aytac@linux.com',
    description='A CUDA extension for custom block operations in PyTorch',
    ext_modules=[
        CUDAExtension(
            name='cuda_block_ops',
            sources=['block_ops.cu'],
            extra_compile_args={
                'cxx': ['-O2', '-Wall'],  
                'nvcc': ['-O2', '-arch=sm_75']  
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=1.8.0',  
        'numpy',         
    ],
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',  
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Block-Based rconstruction',
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: CUDA',
    ],
    keywords='pytorch cuda extension',
)
