from setuptools import setup
from setuptools.extension import Extension
from Cython.Distutils import build_ext
import platform
import numpy
import glob
import re
import eigency

def RemoveMain(listoffiles):
    return [fn for fn in listoffiles if "Main.cpp" not in fn]

# Optimization flags
copt = {
    "msvc": ["/openmp", "/O2", "/Ot", "/MP"],
    "mingw32" : ["-O3", "-fopenmp"],
    "unix": ["-std=c++11", "-O3", "-fopenmp", "-msse3"]
    }

# OpenMP not supported on OSX
if platform.system() == "Darwin":
    copt["unix"].remove("-fopenmp")
    copt["unix"].append("-stdlib=libc++")

lopt = {"mingw32" : ["-fopenmp"] }

class build_ext_subclass(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        if c in copt:
            for e in self.extensions:
                e.extra_compile_args = copt[c]
        if c in lopt:
            for e in self.extensions:
                e.extra_link_args = lopt[c]
        build_ext.build_extensions(self)

sources = ["NonlinearTMM/src/SecondOrderNLTMM.pyx"] + \
    RemoveMain(glob.glob("NonlinearTMM/src/NonlinearTMMCpp/NonlinearTMMCpp/*.cpp"))
    
include_dirs = [r"NonlinearTMM/src/NonlinearTMMCpp/NonlinearTMMCpp",
                r"NonlinearTMM/src/eigen_3.3.2",
                numpy.get_include()] + \
                eigency.get_includes(include_eigen = False)

ext = Extension("NonlinearTMM._SecondOrderNLTMMCython",
    sources = sources,
    include_dirs = include_dirs,
    language = "c++")

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
    cmdclass = {"build_ext": build_ext_subclass},
    ext_modules = [ext],
    python_requires=">=3.5",
    install_requires=[
        "numpy",
        "scipy",
        "eigency>=2.0.0",
    ],
)
