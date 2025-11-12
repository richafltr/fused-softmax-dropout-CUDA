from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_softmax_dropout",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="fused_softmax_dropout._C",
            sources=[
                "fused_softmax_dropout/bindings.cpp",
                "fused_softmax_dropout/kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    packages=["fused_softmax_dropout"],
    python_requires=">=3.7",
    install_requires=["torch"],
)

