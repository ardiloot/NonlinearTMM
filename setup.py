from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy
import glob
import eigency

cmdclass = {'build_ext': build_ext}

ext = Extension("NonlinearTMM._SecondOrderNLTMMCython",
                sources = ["NonlinearTMM/src/SecondOrderNLTMM.pyx"] + 
                            glob.glob("NonlinearTMM/src/NonlinearTMMCpp/NonlinearTMMCpp/*.cpp"),
                include_dirs = [r"NonlinearTMM/src/NonlinearTMMCpp/NonlinearTMMCpp",
                                r"NonlinearTMM/src/eigen_3.3.2",
                                numpy.get_include()] + 
                                eigency.get_includes(include_eigen = False),
                language = "c++",
                extra_compile_args=["/openmp", "/arch:SSE2", "/O2", "/Ot"],
                extra_link_args=[],
    )

setup(name = "NonlinearTMM",
      version = "1.0.1",
      author = "Ardi Loot",
      author_email = "ardi.loot@outlook.com",
      packages = ["NonlinearTMM"],
      cmdclass = cmdclass,
      ext_modules = [ext],)