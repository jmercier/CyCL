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
cimport numpy as np

cdef extern from "pyerrors.h":
    ctypedef class __builtin__.Exception [object PyBaseExceptionObject]: pass


cdef union param:
        np.npy_byte         byte_value              #1
        np.npy_short        short_value             #3
        np.npy_int          int_value               #7
        np.npy_long         long_value
        np.npy_longlong     longlong_value

        np.npy_ubyte        ubyte_value             #2
        np.npy_ushort       ushort_value            #4
        np.npy_ulong        ulong_value
        np.npy_ulonglong    ulonglong_value

        np.npy_float        float_value             #12
        np.npy_double       double_value            #12

        np.npy_int8         int8_value              #1
        np.npy_int16        int16_value             #4
        np.npy_int32        int32_value             #5
        np.npy_int64        int64_value             #7

        np.npy_uint8        uint8_value             #2
        np.npy_uint16       uint16_value            #4
        np.npy_uint32       uint32_value            #6
        np.npy_uint64       uint64_value            #8

        np.npy_float32      float32_value           #11
        np.npy_float64      float64_value           #12
        np.npy_float128     float128_value          #13

        np.npy_complex64    complex64_value         #14
        np.npy_complex128   complex128_value        #15
        np.npy_complex256   complex256_value        #16

        np.npy_intp         intp_value
        cl_sampler          sampler_value
        cl_mem              mem_value


ctypedef param (*param_converter_fct)(object) except *

cdef struct ptype:
    size_t itemsize
    param_converter_fct fct


DEF BYTE_ID          = 0
DEF UBYTE_ID         = 1
DEF SHORT_ID         = 2
DEF USHORT_ID        = 3
DEF INT32_ID         = 4
DEF UINT32_ID        = 5
DEF INT64_ID         = 6
DEF UINT64_ID        = 7

DEF INTP_ID          = 8

DEF FLOAT32_ID       = 9
DEF FLOAT64_ID       = 10
DEF FLOAT128_ID      = 11

DEF BUFFER_ID        = 12
DEF IMAGE_ID         = 12
DEF SAMPLER_ID       = 13


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
    cdef cl_device_id _device

cdef class CLBuffer(CLObject):
    cdef cl_mem _mem
    cdef unsigned int _offset
    cdef CLContext _context
    cdef void *_host_ptr

cdef class CLMappedBuffer(CLObject):
    cdef CLBuffer _buffer
    cdef void * _address
    cdef CLCommandQueue _command_queue
    cdef bint _mapped

cdef class CLCommandQueue(CLObject):
    cdef cl_command_queue _command_queue
    cdef CLContext _context
    cpdef finish(self)
    cpdef flush(self)


cdef class CLImage(CLBuffer):
    cdef void _getSize(self, size_t size[3])

cdef class CLKernel(CLObject):
    cdef cl_kernel _kernel
    cdef CLProgram _program
    cdef tuple _targs

cdef class CLProgram(CLObject):
    cdef cl_program _program
    cdef CLContext _context

    cdef void _build(self, list options)
    cpdef CLKernel createKernel(self, bytes)
    cpdef list createKernelsInProgram(self)
    cpdef bytes getBuildLog(self, CLDevice)

cdef class CLContext(CLObject):
    cdef cl_context _context
    cdef list _devices
    cdef CLBuffer _createBuffer(self, size_t, size_t, cl_mem_flags)
    cpdef CLImage createImage2D(self, size_t, size_t,cl_channel_order, cl_channel_type)
    cpdef CLImage createImage3D(self, size_t, size_t, size_t)
    cpdef CLProgram createProgramWithSource(self, bytes)
    cdef CLCommandQueue _createCommandQueue(self, CLDevice, cl_command_queue_properties)
    cpdef CLSampler createSampler(self, cl_bool, cl_addressing_mode, cl_filter_mode)


cdef class CLEvent(CLObject):
    cdef cl_event _event


cdef class CLSampler(CLObject):
    cdef cl_sampler _sampler
    cdef CLContext _context


