#!/usr/bin/env python
import os

from config import *

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from mako.template import Template
#print "makoing template"
#for file in ['opencl.pyx.mako']:
#    with open(os.path.join("templates", file)) as input:
#        t = Template(input.read()).render()
#    with open(os.path.join("src/%s" % (file[:-5])), 'w') as output:
#        output.write(t)


extensions = [ Extension("cycl.opencl", ['src/opencl.pyx'],
                         libraries = ["OpenCL"],
                         library_dirs = [opencl_library_dir],
                         include_dirs = [opencl_include_dir, 'src']),
               Extension("cycl.clcommand", ['src/clcommand.pyx'],
                             libraries = ["OpenCL"],
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



