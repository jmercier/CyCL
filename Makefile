.SUFFIXES: templates/%.mako
SHELL = /bin/sh

TEMPLATES_DIR:= templates
TEMPLATES_SRC:= $(wildcard ${TEMPLATES_DIR}/*hpp.mako)
TEMPLATES_OBJ:= $(TEMPLATES_SRC:.mako=)

SRCDIR:= ./src
BUILDDIR:= ./build
RENDERER:= ./scripts/gen_template.py

PROJECT:= cycl

CC:= g++
CXXFLAGS:= -O3 -g -Wall
INCLUDE_DIR:= -I${BUILDDIR}/include

all: templates python_package

test : src/test.cpp templates
	${CC} src/test.cpp ${INCLUDE_DIR} ${CXXFLAGS} -o ${BUILDDIR}/$@
	${BUILDDIR}/$@ > test.txt

clean:
	rm -rf ./build

templates : build/command.pyx build/core.pyx

python_package: build/command.pyx build/core.pyx ${PROJECT}/* src/${PROJECT}
	python setup.py build


build/%.pyx : templates/%.pyx.mako templates/functions.mako LICENSE
	@if test ! -d build ; then mkdir -p build; fi
	$(RENDERER) $< $@

build/%.pyx : templates/%.pyx.mako src/eikonal/%.pxd
	@if test ! -d build/${PROJECT} ; then mkdir -p build/${PROJEC}; fi
	@touch build/${PROJECT}/__init__.py
	@cp src/${PROJECT}/*.pxd build/${PROJECT}
	$(RENDERER) $< > $@

