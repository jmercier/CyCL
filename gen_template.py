#!/usr/bin/env python

import os
from mako.template import Template
from mako.lookup import TemplateLookup
mylookup = TemplateLookup(directories=['./'])

print "makoing template"
for file in ['opencl.pyx', 'command.pyx']:
    print file
    t = Template(filename = os.path.join("templates", file + '.mako'), lookup = mylookup)
    with open(os.path.join("src/%s" % file), 'w') as output:
        output.write(t.render())

