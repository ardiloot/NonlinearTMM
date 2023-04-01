import sys
from setuptools import setup
from setuptools.extension import Extension
import numpy as np
import glob
import eigency
from Cython.Build import cythonize

def RemoveMain(listoffiles):
    return [fn for fn in listoffiles if "Main.cpp" not in fn]


extra_compile_args = []
extra_link_args = []

if sys.platform in ("linux", "darwin"):
    extra_compile_args.extend(["-std=c++11", "-fopenmp", "-msse3"])
    extra_link_args.extend(["-lgomp", "-fopenmp"])
elif sys.platform == "win32":
    extra_compile_args.extend(["/openmp"])

sources = ["NonlinearTMM/src/SecondOrderNLTMM.pyx"] + \
    RemoveMain(glob.glob("NonlinearTMM/src/NonlinearTMMCpp/NonlinearTMMCpp/*.cpp"))
extensions = cythonize([
    Extension(
        "NonlinearTMM._SecondOrderNLTMMCython",
        sources=sources,
        include_dirs=[
            np.get_include(),
            r"NonlinearTMM/src/NonlinearTMMCpp/NonlinearTMMCpp",
            r"NonlinearTMM/src/eigen-3.4.0",
        ] + eigency.get_includes(include_eigen = False),
        language="c++",
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
])

long_description = open("README.md", encoding="utf-8").read()

setup(
    name = "NonlinearTMM",
    description="Nonlinear transfer matrix method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    use_scm_version=True,
    author = "Ardi Loot",
    url = "https://github.com/ardiloot/NonlinearTMM",
    author_email = "ardi.loot@outlook.com",
    packages = ["NonlinearTMM"],
    ext_modules=extensions,
    python_requires=">=3.5",
    install_requires=[
        "numpy",
        "scipy",
        "eigency>=2.0.0",
    ],
    extras_require={
        "dev": ["pyyaml", "pytest", "flake8", "pip-tools", "matplotlib"],
        "test": ["pyyaml", "pytest", "flake8"],
    },
)
