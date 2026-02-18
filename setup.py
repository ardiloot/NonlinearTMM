import sys
import glob

import eigency
import numpy as np
from Cython.Build import cythonize
from setuptools import setup
from setuptools.extension import Extension


def _remove_main(files):
    return [f for f in files if "Main.cpp" not in f]


extra_compile_args = []
extra_link_args = []

if sys.platform in ("linux", "darwin"):
    extra_compile_args.extend(["-std=c++11", "-fopenmp", "-msse3"])
    extra_link_args.extend(["-lgomp", "-fopenmp"])
elif sys.platform == "win32":
    extra_compile_args.extend(["/openmp"])

sources = ["NonlinearTMM/src/SecondOrderNLTMM.pyx"] + _remove_main(
    glob.glob("NonlinearTMM/src/NonlinearTMMCpp/NonlinearTMMCpp/*.cpp")
)

extensions = cythonize(
    [
        Extension(
            "NonlinearTMM._SecondOrderNLTMMCython",
            sources=sources,
        include_dirs=[
            np.get_include(),
            "NonlinearTMM/src/NonlinearTMMCpp/NonlinearTMMCpp",
            "NonlinearTMM/src/eigen-3.4.0",
        ] + eigency.get_includes(include_eigen=False),
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        ),
    ],
    force=True,
)

setup(ext_modules=extensions)
