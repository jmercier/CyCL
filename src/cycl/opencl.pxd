# Copyright (c) 2010 Jean-Pascal Mercier
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
#
from libopencl cimport *

cdef extern from "pyerrors.h":
    ctypedef class __builtin__.Exception [object PyBaseExceptionObject]: pass


cdef class CLObject
cdef class CLBuffer
cdef class CLImage
cdef class CLContext
cdef class CLDevice
cdef class CLProgram
cdef class CLKernel
cdef class CLSampler
cdef class CLEvent
cdef class CLCommandQueue

cdef class CLError(Exception): pass
cdef CLError translateError(cl_int error)

cdef class CLObject: pass

cdef class CLDevice(CLObject):
    cdef cl_device_id       _device

cdef class CLBuffer(CLObject):
    cdef cl_mem             _mem
    cdef unsigned int       _offset
    cdef CLContext          _context
    cdef void *             _host_ptr

cdef class CLMappedBuffer(CLObject):
    cdef CLBuffer           _buffer
    cdef void *             _address
    cdef CLCommandQueue     _command_queue
    cdef bint               _mapped

cdef class CLCommandQueue(CLObject):
    cdef cl_command_queue   _command_queue
    cdef CLContext          _context

cdef class CLImage(CLBuffer): pass

cdef class CLKernel(CLObject):
    cdef cl_kernel          _kernel
    cdef CLProgram          _program
    cdef tuple              _targs

cdef class CLProgram(CLObject):
    cdef cl_program         _program
    cdef CLContext          _context

cdef class CLContext(CLObject):
    cdef cl_context         _context
    cdef list               _devices


cdef class CLEvent(CLObject):
    cdef cl_event           _event
    cdef CLCommandQueue     _queue


cdef class CLSampler(CLObject):
    cdef cl_sampler         _sampler
    cdef CLContext          _context

cdef class CLPlatform(CLObject):
    cdef cl_platform_id     _platform

