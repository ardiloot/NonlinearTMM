from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy
import glob
import re
import eigency

def RemoveMain(listoffiles):
    return [fn for fn in listoffiles if "Main.cpp" not in fn]

# Optimization flags
copt = {"msvc": ["/openmp", "/arch:SSE2", "/O2", "/Ot", "/MP"],
         "mingw32" : ["-O3", "-fopenmp"]}
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

# Parse version
__version__ = re.search(r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    open('NonlinearTMM/__init__.py').read()).group(1)

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

setup(name = "NonlinearTMM",
      version = __version__,
      author = "Ardi Loot",
      url = "https://github.com/ardiloot/NonlinearTMM",
      author_email = "ardi.loot@outlook.com",
      packages = ["NonlinearTMM"],
      cmdclass = {"build_ext": build_ext_subclass},
      ext_modules = [ext],)
