#!/usr/bin/env python
import os

from config import *

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

from mako.template import Template
print "building templates"
for file in os.listdir("templates"):
    with open(os.path.join("templates", file)) as input:
        t = Template(input.read()).render()
    with open(os.path.join("src/%s" % (file[:-5])), 'w') as output:
        output.write(t)


opencl_srcs = ['src/opencl.pyx']
opencl_extension = Extension("cycl.opencl", opencl_srcs,
                             libraries = ["OpenCL"],
                             library_dirs = [opencl_library_dir],
                             include_dirs = [opencl_include_dir, 'src'])

cycl_srcs = ['cycl']

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [opencl_extension],
    packages = ['cycl', 'cycl/defines'],
    name = 'CyCL',
    version = '0.0.1',
    description = "Cython Bindings for OpenCL",
    author = "J-Pascal Mercier",
    author_email = "jp.mercier@gmail.com",
    license = "MIT"
)



