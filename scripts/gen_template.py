#!/usr/bin/env python

import os, sys
from mako.template import Template
from mako.lookup import TemplateLookup
mylookup = TemplateLookup(directories=['./'])

with open('LICENSE') as f:
    license_text = f.read()

inp = sys.argv[1]
out = sys.argv[2]

t = Template(filename = inp, lookup = mylookup)
with open(out, 'w') as output:
    output.write(t.render(license = license_text))

