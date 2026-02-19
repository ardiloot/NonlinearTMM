import glob
import platform
import sys

import eigency
import numpy as np
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension

extra_compile_args = []
extra_link_args = []

if sys.platform in ("linux", "darwin"):
    extra_compile_args.extend(["-std=c++17", "-fopenmp"])
    if platform.machine() in ("x86_64", "AMD64", "i686", "i386"):
        extra_compile_args.append("-msse3")
    extra_link_args.extend(["-lgomp", "-fopenmp"])
elif sys.platform == "win32":
    extra_compile_args.extend(["/std:c++17", "/openmp"])

sources = ["NonlinearTMM/src/SecondOrderNLTMM.pyx"] + glob.glob("NonlinearTMM/src/cpp/*.cpp")

extensions = cythonize(
    [
        Extension(
            "NonlinearTMM._SecondOrderNLTMMCython",
            sources=sources,
            include_dirs=[
                np.get_include(),
                "NonlinearTMM/src/cpp",
                "third_party/eigen",
            ]
            + eigency.get_includes(include_eigen=False),
            language="c++",
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
)

setup(ext_modules=extensions)
