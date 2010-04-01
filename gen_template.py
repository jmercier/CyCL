#!/usr/bin/env python

import os
from mako.template import Template

print "makoing template"
for file in ['opencl.pyx.mako']:
    print file
    with open(os.path.join("templates", file)) as input:
        t = Template(input.read()).render()
    with open(os.path.join("src/%s" % (file[:-5])), 'w') as output:
        output.write(t)

