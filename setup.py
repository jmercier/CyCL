#!/usr/bin/python -B
from config import *

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


extensions = [ Extension("cycl.opencl", ['src/opencl.pyx'],
                         libraries = [opencl_library],
                         library_dirs = [opencl_library_dir],
                         include_dirs = [opencl_include_dir, 'src']),
               Extension("cycl.command", ['src/command.pyx'],
                             libraries = [opencl_library],
                             library_dirs = [opencl_library_dir],
                             include_dirs = [opencl_include_dir, 'src'])]

cycl_srcs = ['cycl']

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions,
    packages = ['cycl', 'cycl/defines'],
    name = 'CyCL',
    version = '0.0.1',
    description = "Cython Bindings for OpenCL",
    author = "J-Pascal Mercier",
    author_email = "jp.mercier@gmail.com",
    license = "MIT"
)



