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

cimport numpy as cnp

cdef extern from "pyerrors.h":
    ctypedef class __builtin__.Exception [object PyBaseExceptionObject]: pass

cdef extern from "Python.h":
    ctypedef long long Py_intptr_t



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
    cdef cl_device_id               _device

cdef class CLBuffer(CLObject):
    cdef cl_mem                     _mem
    cdef cl_uint                    _offset
    cdef readonly CLContext         _context

cdef class CLTypedBuffer(CLBuffer):
    cdef readonly cnp.dtype         dtype

cdef class CLMappedBuffer:
    cdef readonly CLBuffer          _buffer
    cdef void *                     _address
    cdef bint                       _ready

cdef class CLCommandQueue(CLObject):
    cdef cl_command_queue           _command_queue
    cdef readonly CLContext         _context

cdef class CLImage(CLBuffer): pass

cdef class CLKernel(CLObject):
    cdef cl_kernel                  _kernel
    cdef readonly CLProgram         _program
    cdef tuple                      _targs
    cdef cl_bool                    _ready

cdef class CLProgram(CLObject):
    cdef cl_program                 _program
    cdef readonly CLContext         _context

cdef class CLContext(CLObject):
    cdef cl_context                 _context
    cdef readonly list              _devices


cdef class CLEvent(CLObject):
    cdef cl_event                   _event
    cdef readonly CLCommandQueue    _queue


cdef class CLSampler(CLObject):
    cdef cl_sampler                 _sampler
    cdef readonly CLContext         _context

cdef class CLPlatform(CLObject):
    cdef cl_platform_id             _platform


cdef class CLCommand:
    cdef object call(self, CLCommandQueue queue)


cdef inline CLEvent _createCLEvent(cl_event event, CLCommandQueue queue):
    cdef CLEvent instance = CLEvent.__new__(CLEvent)
    instance._event             = event
    instance._queue             = queue
    return instance

cdef inline CLPlatform _createCLPlatform(cl_platform_id pid):
    cdef CLPlatform instance    = CLPlatform.__new__(CLPlatform)
    instance._platform          = pid
    return instance

cdef inline CLDevice _createCLDevice(cl_device_id did):
    cdef CLDevice instance      = CLDevice.__new__(CLDevice)
    instance._device            = did
    return instance

cdef inline CLImage _createCLImage(cl_mem mem, CLContext context, cl_uint offset):
    cdef CLImage instance       = CLImage.__new__(CLImage)
    instance._mem               = mem
    instance._context           = context
    instance._offset            = offset
    return instance

cdef inline CLBuffer _createCLBuffer(cl_mem mem, CLContext context, cl_uint offset):
    cdef CLBuffer instance       = CLBuffer.__new__(CLBuffer)
    instance._mem               = mem
    instance._context           = context
    instance._offset            = offset
    return instance

cdef inline CLBuffer _createCLTypedBuffer(cl_mem mem, CLContext context, cl_uint offset, cnp.dtype dtype):
    cdef CLTypedBuffer instance = CLTypedBuffer.__new__(CLTypedBuffer)
    instance._mem               = mem
    instance._context           = context
    instance._offset            = offset
    instance.dtype              = dtype
    return instance

cdef inline CLCommandQueue _createCLCommandQueue(CLContext context, cl_command_queue queue):
    cdef CLCommandQueue instance = CLCommandQueue.__new__(CLCommandQueue)
    instance._context           = context
    instance._command_queue     = queue
    return instance

cdef inline CLSampler _createCLSampler(CLContext context, cl_sampler sampler):
    cdef CLSampler instance = CLSampler.__new__(CLSampler)
    instance._context           = context
    instance._sampler           = sampler
    return instance

cdef inline CLProgram _createCLProgram(CLContext context, cl_program program):
    cdef CLProgram instance = CLProgram.__new__(CLProgram)
    instance._context           = context
    instance._program           = program
    return instance

cdef inline CLKernel _createCLKernel(CLProgram program, cl_kernel kernel):
    cdef CLKernel instance      = CLKernel.__new__(CLKernel)
    instance._program           = program
    instance._kernel            = kernel
    instance._ready             = False
    return instance

cdef inline CLContext _createCLContext(list device, cl_context context):
    cdef CLContext instance     = CLContext.__new__(CLContext)
    instance._context           = context
    instance._devices            = device
    return instance


