#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from configs import *

opencl_srcs = ['src/opencl.pyx']
opencl_extension = Extension("cycl/opencl", opencl_srcs,
                             libraries = ["OpenCL"],
                             library_dirs = [opencl_library_dir],
                             include_dirs = [opencl_include_dir])

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [opencl_extension],
    name = 'CyCL',
    version = '0.0.1',
    description = "Cython Bindings for OpenCL",
    author = "J-Pascal Mercier",
    author_email = "jp.mercier@gmail.com",
    license = "MIT"
)



