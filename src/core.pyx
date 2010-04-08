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

cdef dict error_translation_table = {
        CL_SUCCESS                               : "CL_SUCCESS",
        CL_DEVICE_NOT_FOUND                      : "CL_DEVICE_NOT_FOUND",
        CL_DEVICE_NOT_AVAILABLE                  : "CL_DEVICE_NOT_AVAILABLE",
        CL_COMPILER_NOT_AVAILABLE                : "CL_COMPILER_NOT_AVAILABLE",
        CL_MEM_OBJECT_ALLOCATION_FAILURE         : "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        CL_OUT_OF_RESOURCES                      : "CL_OUT_OF_RESOURCES",
        CL_OUT_OF_HOST_MEMORY                    : "CL_OUT_OF_HOST_MEMORY",
        CL_PROFILING_INFO_NOT_AVAILABLE          : "CL_PROFILING_INFO_NOT_AVAILABLE",
        CL_MEM_COPY_OVERLAP                      : "CL_MEM_COPY_OVERLAP",
        CL_IMAGE_FORMAT_MISMATCH                 : "CL_IMAGE_FORMAT_MISMATCH",
        CL_IMAGE_FORMAT_NOT_SUPPORTED            : "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        CL_BUILD_PROGRAM_FAILURE                 : "CL_BUILD_PROGRAM_FAILURE",
        CL_MAP_FAILURE                           : "CL_MAP_FAILURE",
        CL_INVALID_VALUE                         : "CL_INVALID_VALUE",
        CL_INVALID_DEVICE_TYPE                   : "CL_INVALID_DEVICE_TYPE",
        CL_INVALID_PLATFORM                      : "CL_INVALID_PLATFORM",
        CL_INVALID_DEVICE                        : "CL_INVALID_DEVICE",
        CL_INVALID_CONTEXT                       : "CL_INVALID_CONTEXT",
        CL_INVALID_QUEUE_PROPERTIES              : "CL_INVALID_QUEUE_PROPERTIES",
        CL_INVALID_COMMAND_QUEUE                 : "CL_INVALID_COMMAND_QUEUE",
        CL_INVALID_HOST_PTR                      : "CL_INVALID_HOST_PTR",
        CL_INVALID_MEM_OBJECT                    : "CL_INVALID_MEM_OBJECT",
        CL_INVALID_IMAGE_FORMAT_DESCRIPTOR       : "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        CL_INVALID_IMAGE_SIZE                    : "CL_INVALID_IMAGE_SIZE",
        CL_INVALID_SAMPLER                       : "CL_INVALID_SAMPLER",
        CL_INVALID_BINARY                        : "CL_INVALID_BINARY",
        CL_INVALID_BUILD_OPTIONS                 : "CL_INVALID_BUILD_OPTIONS",
        CL_INVALID_PROGRAM                       : "CL_INVALID_PROGRAM",
        CL_INVALID_PROGRAM_EXECUTABLE            : "CL_INVALID_PROGRAM_EXECUTABLE",
        CL_INVALID_KERNEL_NAME                   : "CL_INVALID_KERNEL_NAME",
        CL_INVALID_KERNEL_DEFINITION             : "CL_INVALID_KERNEL_DEFINITION",
        CL_INVALID_KERNEL                        : "CL_INVALID_KERNEL",
        CL_INVALID_ARG_INDEX                     : "CL_INVALID_ARG_INDEX",
        CL_INVALID_ARG_VALUE                     : "CL_INVALID_ARG_VALUE",
        CL_INVALID_ARG_SIZE                      : "CL_INVALID_ARG_SIZE",
        CL_INVALID_KERNEL_ARGS                   : "CL_INVALID_KERNEL_ARGS",
        CL_INVALID_WORK_DIMENSION                : "CL_INVALID_WORK_DIMENSION",
        CL_INVALID_WORK_GROUP_SIZE               : "CL_INVALID_WORK_GROUP_SIZE",
        CL_INVALID_WORK_ITEM_SIZE                : "CL_INVALID_WORK_ITEM_SIZE",
        CL_INVALID_GLOBAL_OFFSET                 : "CL_INVALID_GLOBAL_OFFSET",
        CL_INVALID_EVENT_WAIT_LIST               : "CL_INVALID_EVENT_WAIT_LIST",
        CL_INVALID_EVENT                         : "CL_INVALID_EVENT",
        CL_INVALID_OPERATION                     : "CL_INVALID_OPERATION",
        CL_INVALID_GL_OBJECT                     : "CL_INVALID_GL_OBJECT",
        CL_INVALID_BUFFER_SIZE                   : "CL_INVALID_BUFFER_SIZE",
        CL_INVALID_MIP_LEVEL                     : "CL_INVALID_MIP_LEVEL",
        CL_INVALID_GLOBAL_WORK_SIZE              : "CL_INVALID_GLOBAL_WORK_SIZE",
}


cdef CLError translateError(cl_int errcode):
    return CLError(error_translation_table[errcode])


#
#
#   Args Translation
#
#

cdef union param:
    cl_mem              mem_value
    cl_sampler          sampler_value
    cl_char         char_value
    cl_uchar        uchar_value
    cl_short        short_value
    cl_ushort       ushort_value
    cl_int          int_value
    cl_uint         uint_value
    cl_long         long_value
    cl_ulong        ulong_value
    cl_half         half_value
    cl_float        float_value
    cl_double       double_value
    cl_bool         bool_value

ctypedef param (*param_converter_fct)(object) except *

cdef struct ptype:
    size_t                  itemsize
    param_converter_fct     fct

DEF MAX_ARG_TRANSLATION = 14
cdef ptype param_converter_array[MAX_ARG_TRANSLATION]

cdef param from_char(object val) except *:
    cdef param p
    p.char_value = <cl_char>val
    return p
param_converter_array[0].itemsize = sizeof(cl_char)
param_converter_array[0].fct = from_char

cdef param from_uchar(object val) except *:
    cdef param p
    p.uchar_value = <cl_uchar>val
    return p
param_converter_array[1].itemsize = sizeof(cl_uchar)
param_converter_array[1].fct = from_uchar

cdef param from_short(object val) except *:
    cdef param p
    p.short_value = <cl_short>val
    return p
param_converter_array[2].itemsize = sizeof(cl_short)
param_converter_array[2].fct = from_short

cdef param from_ushort(object val) except *:
    cdef param p
    p.ushort_value = <cl_ushort>val
    return p
param_converter_array[3].itemsize = sizeof(cl_ushort)
param_converter_array[3].fct = from_ushort

cdef param from_int(object val) except *:
    cdef param p
    p.int_value = <cl_int>val
    return p
param_converter_array[4].itemsize = sizeof(cl_int)
param_converter_array[4].fct = from_int

cdef param from_uint(object val) except *:
    cdef param p
    p.uint_value = <cl_uint>val
    return p
param_converter_array[5].itemsize = sizeof(cl_uint)
param_converter_array[5].fct = from_uint

cdef param from_long(object val) except *:
    cdef param p
    p.long_value = <cl_long>val
    return p
param_converter_array[6].itemsize = sizeof(cl_long)
param_converter_array[6].fct = from_long

cdef param from_ulong(object val) except *:
    cdef param p
    p.ulong_value = <cl_ulong>val
    return p
param_converter_array[7].itemsize = sizeof(cl_ulong)
param_converter_array[7].fct = from_ulong

cdef param from_half(object val) except *:
    cdef param p
    p.half_value = <cl_half>val
    return p
param_converter_array[8].itemsize = sizeof(cl_half)
param_converter_array[8].fct = from_half

cdef param from_float(object val) except *:
    cdef param p
    p.float_value = <cl_float>val
    return p
param_converter_array[9].itemsize = sizeof(cl_float)
param_converter_array[9].fct = from_float

cdef param from_double(object val) except *:
    cdef param p
    p.double_value = <cl_double>val
    return p
param_converter_array[10].itemsize = sizeof(cl_double)
param_converter_array[10].fct = from_double

cdef param from_bool(object val) except *:
    cdef param p
    p.bool_value = <cl_bool>val
    return p
param_converter_array[11].itemsize = sizeof(cl_bool)
param_converter_array[11].fct = from_bool

cdef param from_CLBuffer(object val) except *:
    cdef CLBuffer buf_val = val
    cdef param p
    p.mem_value = buf_val._mem
    return p
param_converter_array[12].itemsize = sizeof(cl_mem)
param_converter_array[12].fct = from_CLBuffer

cdef param from_CLSampler(object val) except *:
    cdef CLSampler buf_val = val
    cdef param p
    p.sampler_value = buf_val._sampler
    return p
param_converter_array[13].itemsize = sizeof(cl_sampler)
param_converter_array[13].fct = from_CLSampler

class parameter_type(CLObject):
    CHAR_TYPE          = 0
    UCHAR_TYPE         = 1
    SHORT_TYPE         = 2
    USHORT_TYPE        = 3
    INT_TYPE           = 4
    UINT_TYPE          = 5
    LONG_TYPE          = 6
    ULONG_TYPE         = 7
    HALF_TYPE          = 8
    FLOAT_TYPE         = 9
    DOUBLE_TYPE        = 10
    BOOL_TYPE          = 11
    MEM_TYPE           = 12
    SAMPLER_TYPE       = 13
    IMAGE_TYPE         = MEM_TYPE

class channel_type(CLObject):
    SNORM_INT8                        = CL_SNORM_INT8
    SNORM_INT16                       = CL_SNORM_INT16
    UNORM_INT8                        = CL_UNORM_INT8
    UNORM_INT16                       = CL_UNORM_INT16
    UNORM_SHORT_565                   = CL_UNORM_SHORT_565
    UNORM_SHORT_555                   = CL_UNORM_SHORT_555
    UNORM_INT_101010                  = CL_UNORM_INT_101010
    SIGNED_INT8                       = CL_SIGNED_INT8
    SIGNED_INT16                      = CL_SIGNED_INT16
    SIGNED_INT32                      = CL_SIGNED_INT32
    UNSIGNED_INT8                     = CL_UNSIGNED_INT8
    UNSIGNED_INT16                    = CL_UNSIGNED_INT16
    UNSIGNED_INT32                    = CL_UNSIGNED_INT32
    HALF_FLOAT                        = CL_HALF_FLOAT
    FLOAT                             = CL_FLOAT

class memory_flag(CLObject):
    MEM_READ_WRITE                    = CL_MEM_READ_WRITE
    MEM_WRITE_ONLY                    = CL_MEM_WRITE_ONLY
    MEM_READ_ONLY                     = CL_MEM_READ_ONLY
    MEM_USE_HOST_PTR                  = CL_MEM_USE_HOST_PTR
    MEM_ALLOC_HOST_PTR                = CL_MEM_ALLOC_HOST_PTR
    MEM_COPY_HOST_PTR                 = CL_MEM_COPY_HOST_PTR

class command_queue_flag(CLObject):
    QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE  = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
    QUEUE_PROFILING_ENABLE            = CL_QUEUE_PROFILING_ENABLE

class addressing_mode(CLObject):
    ADDRESS_NONE                      = CL_ADDRESS_NONE
    ADDRESS_CLAMP_TO_EDGE             = CL_ADDRESS_CLAMP_TO_EDGE
    ADDRESS_CLAMP                     = CL_ADDRESS_CLAMP
    ADDRESS_REPEAT                    = CL_ADDRESS_REPEAT

class device_type(CLObject):
    DEVICE_TYPE_DEFAULT               = CL_DEVICE_TYPE_DEFAULT
    DEVICE_TYPE_CPU                   = CL_DEVICE_TYPE_CPU
    DEVICE_TYPE_GPU                   = CL_DEVICE_TYPE_GPU
    DEVICE_TYPE_ACCELERATOR           = CL_DEVICE_TYPE_ACCELERATOR
    DEVICE_TYPE_ALL                   = CL_DEVICE_TYPE_ALL

class channel_order(CLObject):
    R                                 = CL_R
    A                                 = CL_A
    RG                                = CL_RG
    RA                                = CL_RA
    RGB                               = CL_RGB
    RGBA                              = CL_RGBA
    BGRA                              = CL_BGRA
    ARGB                              = CL_ARGB
    INTENSITY                         = CL_INTENSITY
    LUMINANCE                         = CL_LUMINANCE

class event_status(CLObject):
    COMPLETE                          = CL_COMPLETE
    RUNNING                           = CL_RUNNING
    SUBMITTED                         = CL_SUBMITTED
    QUEUED                            = CL_QUEUED

class command_type(CLObject):
    COMMAND_NDRANGE_KERNEL            = CL_COMMAND_NDRANGE_KERNEL
    COMMAND_TASK                      = CL_COMMAND_TASK
    COMMAND_NATIVE_KERNEL             = CL_COMMAND_NATIVE_KERNEL
    COMMAND_READ_BUFFER               = CL_COMMAND_READ_BUFFER
    COMMAND_WRITE_BUFFER              = CL_COMMAND_WRITE_BUFFER
    COMMAND_COPY_BUFFER               = CL_COMMAND_COPY_BUFFER
    COMMAND_READ_IMAGE                = CL_COMMAND_READ_IMAGE
    COMMAND_WRITE_IMAGE               = CL_COMMAND_WRITE_IMAGE
    COMMAND_COPY_IMAGE                = CL_COMMAND_COPY_IMAGE
    COMMAND_COPY_IMAGE_TO_BUFFER      = CL_COMMAND_COPY_IMAGE_TO_BUFFER
    COMMAND_COPY_BUFFER_TO_IMAGE      = CL_COMMAND_COPY_BUFFER_TO_IMAGE
    COMMAND_MAP_BUFFER                = CL_COMMAND_MAP_BUFFER
    COMMAND_MAP_IMAGE                 = CL_COMMAND_MAP_IMAGE
    COMMAND_UNMAP_MEM_OBJECT          = CL_COMMAND_UNMAP_MEM_OBJECT
    COMMAND_MARKER                    = CL_COMMAND_MARKER
    COMMAND_ACQUIRE_GL_OBJECTS        = CL_COMMAND_ACQUIRE_GL_OBJECTS
    COMMAND_RELEASE_GL_OBJECTS        = CL_COMMAND_RELEASE_GL_OBJECTS

class filter_mode(CLObject):
    FILTER_NEAREST                    = CL_FILTER_NEAREST
    FILTER_LINEAR                     = CL_FILTER_LINEAR




#
#
#   Classes
#
#

cdef class CLCommandQueue(CLObject):
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseCommandQueue(self._command_queue)
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % \
                                self.__class__.__name__)

    def flush(self):
        cdef cl_int errcode = clFlush(self._command_queue)
        if errcode < 0: raise CLError(error_translation_table[errcode])

    def finish(self):
        cdef cl_int errcode = clFinish(self._command_queue)
        if errcode < 0: raise CLError(error_translation_table[errcode])

    def enqueue(self, CLCommand cmd):
        return cmd.call(self)

    def enqueueWaitForEvents(self, list events):
        cdef cl_event lst[100]
        cdef CLEvent evt
        cdef int num_events = min(100, len(events))
        cdef cl_int errcode
        for i from 0 <= i < num_events:
            evt = events[i]
            lst[i] = evt._event
        errcode = clEnqueueWaitForEvents(self._command_queue, num_events, lst)
        if errcode < 0: raise CLError(error_translation_table[errcode])

cdef class CLProgram(CLObject):
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseProgram(self._program)
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % \
                                self.__class__.__name__)

    def createKernelsInProgram(self):
        cdef cl_kernel kernels[20]
        cdef cl_uint num_kernels
        cdef cl_int errcode = clCreateKernelsInProgram(self._program, 20,
                                                       kernels, &num_kernels)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        return [_createCLKernel(self, kernels[i]) for i from 0<= i < num_kernels]

    def createKernel(self, bytes string):
        cdef cl_int errcode
        cdef cl_kernel kernel = clCreateKernel(self._program, string, &errcode)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        return _createCLKernel(self, kernel)

    def getBuildLog(self, CLDevice device):
        cdef char log[10000]
        cdef size_t size
        cdef cl_int errcode
        errcode = clGetProgramBuildInfo(self._program, device._device,
                                        CL_PROGRAM_BUILD_LOG,
                                        10000, log, &size)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        s = log[:size]
        return s

    def build(self, bytes options = ""):
        cdef char *coptions = options
        cdef cl_int errcode = clBuildProgram(self._program,
                                             0, NULL,
                                             coptions, NULL, NULL)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        return self

cdef class CLMappedBuffer:
    def __dealloc__(self):
        if self._ready: print "Memory leak detected, UNMAP IS MENDATORY"

    def __cinit__(self):
        self._ready = False

    def __repr__(self):
        return '<%s address="%s" mapped="%s">' %  \
                (self.__class__.__name__, self.address, self.mapped, )
    property address:
        def __get__(self):
            return <Py_intptr_t> self._address

    property __array_interface__:
        def __get__(self):
            if not self._ready: raise AttributeError ("Memory not Mapped")
            return { "shape"        : (self._buffer.size,),
                     "typestr"      : "|i1",
                     "data"         : (<Py_intptr_t> self._address, False),
                     "version"      : 3}

    property mapped:
        def __get__(self):
            return self._ready

    property size:
        def __get__(self):
            if not self._ready: raise AttributeError ("Memory not Mapped")
            return self._buffer.size



cdef class CLDevice(CLObject):
    
    property maxWorkGroupSize:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef size_t result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_MAX_WORK_GROUP_SIZE,
                                  sizeof(size_t), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property profilingTimerResolution:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef size_t result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                                  sizeof(size_t), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property image2DMaxSize:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef size_t r_0
            cdef size_t r_1
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                                  sizeof(size_t), &r_0, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_IMAGE2D_MAX_WIDTH,
                                  sizeof(size_t), &r_1, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return (r_0, r_1, )

    property image3DMaxSize:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef size_t r_0
            cdef size_t r_1
            cdef size_t r_2
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                                  sizeof(size_t), &r_0, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_IMAGE3D_MAX_WIDTH,
                                  sizeof(size_t), &r_1, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_IMAGE3D_MAX_DEPTH,
                                  sizeof(size_t), &r_2, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return (r_0, r_1, r_2, )


    property globalMemSize:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_ulong result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_GLOBAL_MEM_SIZE,
                                  sizeof(cl_ulong), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property globalMemCacheSize:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_ulong result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                                  sizeof(cl_ulong), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property globalMemCachelineSize:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_ulong result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                                  sizeof(cl_ulong), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property maxConstantBufferSize:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_ulong result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                                  sizeof(cl_ulong), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property maxMemAllocSize:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_ulong result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                                  sizeof(cl_ulong), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property type:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_ulong result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_TYPE,
                                  sizeof(cl_ulong), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result


    property driverVersion:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef bytes result
            cdef char char_result[256]
            errcode = clGetDeviceInfo(self._device,
                                  CL_DRIVER_VERSION,
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
            return result

    property vendor:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef bytes result
            cdef char char_result[256]
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_VERSION,
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
            return result

    property version:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef bytes result
            cdef char char_result[256]
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_VENDOR,
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
            return result

    property profile:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef bytes result
            cdef char char_result[256]
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_PROFILE,
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
            return result

    property name:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef bytes result
            cdef char char_result[256]
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_NAME,
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
            return result

    property extensions:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef bytes result
            cdef char char_result[256]
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_EXTENSIONS,
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
            return result


    property imageSupport:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_bool result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_IMAGE_SUPPORT,
                                  sizeof(cl_bool), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property ECCSupport:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_bool result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_ERROR_CORRECTION_SUPPORT,
                                  sizeof(cl_bool), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property endianLittle:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_bool result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_ENDIAN_LITTLE,
                                  sizeof(cl_bool), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property compilerAvailable:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_bool result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_COMPILER_AVAILABLE,
                                  sizeof(cl_bool), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property available:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_bool result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_AVAILABLE,
                                  sizeof(cl_bool), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result


    property addressBits:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_ADDRESS_BITS,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property vendorId:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_VENDOR_ID,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property maxClockFrequency:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_MAX_CLOCK_FREQUENCY,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property maxComputeUnits:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_MAX_COMPUTE_UNITS,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property maxWorkItemDimensions:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property maxConstantArgs:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_MAX_CONSTANT_ARGS,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property minDataTypeAlignSize:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property maxWriteImageArgs:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property memBaseAddrAlign:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property preferredVectorWidthChar:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property preferredVectorWidthShort:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property preferredVectorWidthInt:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property preferredVectorWidthLong:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property preferredVectorWidthFloat:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property preferredVectorWidthDouble:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetDeviceInfo(self._device,
                                  CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result


    def __repr__(self):
        return '<%s name="%s" type="%s" version="%s">' %  \
                (self.__class__.__name__, self.name, self.type, self.version, )


cdef class CLPlatform(CLObject):
    
    property version:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef bytes result
            cdef char char_result[256]
            errcode = clGetPlatformInfo(self._platform,
                                  CL_PLATFORM_VERSION,
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
            return result

    property name:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef bytes result
            cdef char char_result[256]
            errcode = clGetPlatformInfo(self._platform,
                                  CL_PLATFORM_NAME,
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
            return result

    property vendor:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef bytes result
            cdef char char_result[256]
            errcode = clGetPlatformInfo(self._platform,
                                  CL_PLATFORM_VENDOR,
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
            return result

    property extensions:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef bytes result
            cdef char char_result[256]
            errcode = clGetPlatformInfo(self._platform,
                                  CL_PLATFORM_EXTENSIONS,
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
            return result

    property profile:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef bytes result
            cdef char char_result[256]
            errcode = clGetPlatformInfo(self._platform,
                                  CL_PLATFORM_PROFILE,
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
            return result


    def __repr__(self):
        return '<%s name="%s" vendor="%s" version="%s">' %  \
                (self.__class__.__name__, self.name, self.vendor, self.version, )
    def getDevices(self, cl_device_type dtype = 0xFFFFFFFF):
        cdef cl_device_id devices[10]
        cdef cl_uint num_devices
        cdef cl_int errcode = clGetDeviceIDs(self._platform, dtype, 10,
                                             devices, &num_devices)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        return [_createCLDevice(devices[i]) for i from 0 <= i < num_devices]

    def createContext(self, list devices, tuple properties = ()):
        """
        Creates an OpenCL context.
        """
        cdef long num_devices = len(devices)
        cdef cl_device_id clDevices[100]
        cdef cl_context_properties cproperties[100]
        cdef cl_int errcode
        cproperties[0] = CL_CONTEXT_PLATFORM
        cproperties[1] = <cl_context_properties>self._platform
        cdef tuple prop
        cdef size_t num_properties = len(properties)
        for i from 0 < i <= min(100, num_properties):
            prop = properties[i - 1]
            cproperties[i * 2] = <cl_context_properties>prop[0]
            cproperties[i * 2 + 1] = <cl_context_properties>prop[1]
        properties[(num_properties + 1) * 2] = <cl_context_properties>0
        for i from 0 <= i < min(num_devices, 100):
            clDevices[i] = (<CLDevice>devices[i])._device
        cdef cl_context context = clCreateContext(cproperties, num_devices, clDevices,
                                              NULL, NULL, &errcode)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        return _createCLContext(devices, context)


cdef class CLBuffer(CLObject):
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseMemObject(self._mem)
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % \
                                self.__class__.__name__)

    
    property size:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef size_t result
            errcode = clGetMemObjectInfo(self._mem,
                                  CL_MEM_SIZE,
                                  sizeof(size_t), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result


    def __repr__(self):
        return '<%s size="%s" offset="%s">' %  \
                (self.__class__.__name__, self.size, self.offset, )
    property offset:
        def __get__(self):
            return self._offset


cdef class CLImage(CLBuffer):
    
    property slicePitch:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef size_t result
            errcode = clGetImageInfo(self._mem,
                                  CL_IMAGE_SLICE_PITCH,
                                  sizeof(size_t), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property elementSize:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef size_t result
            errcode = clGetImageInfo(self._mem,
                                  CL_IMAGE_ELEMENT_SIZE,
                                  sizeof(size_t), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property shape:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef size_t r_0
            cdef size_t r_1
            cdef size_t r_2
            errcode = clGetImageInfo(self._mem,
                                  CL_IMAGE_WIDTH,
                                  sizeof(size_t), &r_0, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            errcode = clGetImageInfo(self._mem,
                                  CL_IMAGE_HEIGHT,
                                  sizeof(size_t), &r_1, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            errcode = clGetImageInfo(self._mem,
                                  CL_IMAGE_DEPTH,
                                  sizeof(size_t), &r_2, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return (r_0, r_1, r_2, )

    property rowPitch:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef size_t result
            errcode = clGetImageInfo(self._mem,
                                  CL_IMAGE_ROW_PITCH,
                                  sizeof(size_t), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result


    def __repr__(self):
        return '<%s shape="%s">' %  \
                (self.__class__.__name__, self.shape, )


cdef class CLKernel(CLObject):
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseKernel(self._kernel)
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % \
                                self.__class__.__name__)

    
    property name:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef bytes result
            cdef char char_result[256]
            errcode = clGetKernelInfo(self._kernel,
                                  CL_KERNEL_FUNCTION_NAME,
                                  256 * sizeof(char), char_result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            result = char_result[:size - 1]
            return result


    property numArgs:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetKernelInfo(self._kernel,
                                  CL_KERNEL_NUM_ARGS,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result


    def __repr__(self):
        return '<%s name="%s" numArgs="%s" ready="%s">' %  \
                (self.__class__.__name__, self.name, self.numArgs, self.ready, )
    property parameters:
        def __get__(self):
            return self._targs

        def __set__(self, tuple value):
            cdef unsigned int index
            cdef cl_uint num_args
            cdef cl_int errcode = clGetKernelInfo(self._kernel,
                                      CL_KERNEL_NUM_ARGS,
                                      sizeof(cl_uint), &num_args, NULL)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            for i from 0 <= i < num_args:
                index = value[i]
                if index >= MAX_ARG_TRANSLATION:
                    raise AttributeError("Unknown Type")
            self._ready = True
            self._targs = value


    property ready:
        def __get__(self):
            return self._ready

    def setArgs(self, *args):
        cdef unsigned int index
        cdef param p
        cdef cl_int errcode
        if not self._ready: raise AttributeError("Kernel is not ready : did you forget to TYPE it")
        if len(args) != len(self._targs): raise AttributeError("Number Mismatch")
        for i from 0 <= i < len(args):
            index = self._targs[i]
            p = param_converter_array[index].fct(args[i])
            errcode = clSetKernelArg(self._kernel,
                                     i,param_converter_array[index].itemsize,
                                     &p)
            if errcode < 0: raise CLError(error_translation_table[errcode])


cdef class CLEvent(CLObject):
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseEvent(self._event)
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % \
                                self.__class__.__name__)

    
    property type:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_int result
            errcode = clGetEventInfo(self._event,
                                  CL_EVENT_COMMAND_TYPE,
                                  sizeof(cl_int), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property status:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_int result
            errcode = clGetEventInfo(self._event,
                                  CL_EVENT_COMMAND_EXECUTION_STATUS,
                                  sizeof(cl_int), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result


    
    property profilingQueued:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_ulong result
            errcode = clGetEventProfilingInfo(self._event,
                                  CL_PROFILING_COMMAND_QUEUED,
                                  sizeof(cl_ulong), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property profilingSubmit:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_ulong result
            errcode = clGetEventProfilingInfo(self._event,
                                  CL_PROFILING_COMMAND_SUBMIT,
                                  sizeof(cl_ulong), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property profilingStart:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_ulong result
            errcode = clGetEventProfilingInfo(self._event,
                                  CL_PROFILING_COMMAND_START,
                                  sizeof(cl_ulong), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property profilingEnd:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_ulong result
            errcode = clGetEventProfilingInfo(self._event,
                                  CL_PROFILING_COMMAND_END,
                                  sizeof(cl_ulong), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result


    def __repr__(self):
        return '<%s type="%s" status="%s">' %  \
                (self.__class__.__name__, self.type, self.status, )


cdef class CLSampler(CLObject):
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseSampler(self._sampler)
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % \
                                self.__class__.__name__)

    def __repr__(self):
        return '<%s normalized="%s" filterMode="%s" addressingMode="%s">' %  \
                (self.__class__.__name__, self.normalized, self.filterMode, self.addressingMode, )
    
    property normalized:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetSamplerInfo(self._sampler,
                                  CL_SAMPLER_NORMALIZED_COORDS,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property filterMode:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetSamplerInfo(self._sampler,
                                  CL_SAMPLER_FILTER_MODE,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result

    property addressingMode:
        def __get__(self):
            cdef size_t size
            cdef cl_int errcode
            cdef cl_uint result
            errcode = clGetSamplerInfo(self._sampler,
                                  CL_SAMPLER_ADDRESSING_MODE,
                                  sizeof(cl_uint), &result, &size)
            if errcode < 0: raise CLError(error_translation_table[errcode])
            return result




cdef class CLContext(CLObject):
    """
    This object represent a CLContext.
    """
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseContext (self._context)
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % \
                                self.__class__.__name__)

    def __repr__(self):
        return '<%s devices="%s">' %  \
                (self.__class__.__name__, self.devices, )
    def createBuffer(self, size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE):
        cdef cl_uint offset = 0
        cdef cl_int errcode
        cdef cl_mem mem = clCreateBuffer(self._context,
                                         flags, size,
                                         NULL, &errcode)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        return _createCLBuffer(mem, self, offset)

    def createImage2D(self, size_t width, size_t height,
                      cl_channel_order order, cl_channel_type itype,
                      cl_mem_flags flags = CL_MEM_READ_WRITE ):
        cdef cl_image_format format = [order, itype]
        cdef cl_int errcode
        cdef cl_mem mem = clCreateImage2D(self._context, flags, &format,
                                          width, height,
                                          0, NULL, &errcode)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        return _createCLImage(mem, self, 0)

    def createImage3D(self, size_t width, size_t height, size_t depth,
                      cl_channel_order order, cl_channel_type itype,
                      cl_mem_flags flags = CL_MEM_READ_WRITE):
        cdef cl_image_format format = [order, itype]
        cdef cl_int errcode
        cdef cl_mem mem = clCreateImage3D(self._context, flags, &format,
                                          width, height, depth,
                                          0, 0, NULL, &errcode)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        return _createCLImage(mem, self, 0)

    def createCommandQueue(self, CLDevice device,
                           cl_command_queue_properties flags = <cl_command_queue_properties>0):
        cdef cl_int errcode
        cdef cl_command_queue queue = clCreateCommandQueue(self._context,
                                                           device._device,
                                                           flags, &errcode)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        return _createCLCommandQueue(self, queue)

    def createSampler(self, cl_bool normalized,
                      cl_addressing_mode amode, cl_filter_mode fmode):
        cdef cl_int errcode
        cdef cl_sampler sampler = clCreateSampler(self._context, normalized,
                                                  amode, fmode, &errcode)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        return _createCLSampler(self, sampler)

    def createProgramWithSource(self, bytes pystring):
        cdef const_char_ptr strings[1]
        cdef size_t sizes = len(pystring)
        cdef cl_int errcode
        strings[0] = pystring
        cdef cl_program program = clCreateProgramWithSource(self._context, 1,
                                                            strings, &sizes,
                                                            &errcode)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        return _createCLProgram(self, program)

    property devices:
        def __get__(self):
            return self._devices

cdef class CLCommand:
    cdef object call(self, CLCommandQueue queue):
        raise AttributeError("Abstract Method")

cdef class CLGLBuffer(CLBuffer):
    pass


#
#
#   Module level API
#
#

cpdef list getPlatforms():
    """
    Obtain the list of platforms available.
    """
    cdef cl_platform_id platforms[15]
    cdef cl_uint num_platforms
    cdef cl_int errcode = clGetPlatformIDs(15, platforms, &num_platforms)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return [_createCLPlatform(platforms[i]) for i from 0 <= i < num_platforms]

cpdef waitForEvents(list events):
    """
    Waits on the host thread for commands identified by event objects in
    event_list to complet.
    """
    cdef int num_events = len(events)
    cdef cl_event lst[100]
    for i from 0 <= i < min(num_events, 100):
        lst[i] = (<CLEvent>events[i])._event
    cdef cl_int errcode = clWaitForEvents(num_events, lst)
    if errcode < 0: raise CLError(error_translation_table[errcode])





