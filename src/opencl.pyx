 
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
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

cimport opencl
cimport numpy as np
from opencl cimport *

from defines import *

cdef dict error_translation_table = {
        CL_SUCCESS : "CL_SUCCESS",
        CL_DEVICE_NOT_FOUND : "CL_DEVICE_NOT_FOUND",
        CL_DEVICE_NOT_AVAILABLE : "CL_DEVICE_NOT_AVAILABLE",
        CL_COMPILER_NOT_AVAILABLE : "CL_COMPILER_NOT_AVAILABLE",
        CL_MEM_OBJECT_ALLOCATION_FAILURE : "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        CL_OUT_OF_RESOURCES : "CL_OUT_OF_RESOURCES",
        CL_OUT_OF_HOST_MEMORY : "CL_OUT_OF_HOST_MEMORY",
        CL_PROFILING_INFO_NOT_AVAILABLE : "CL_PROFILING_INFO_NOT_AVAILABLE",
        CL_MEM_COPY_OVERLAP : "CL_MEM_COPY_OVERLAP",
        CL_IMAGE_FORMAT_MISMATCH : "CL_IMAGE_FORMAT_MISMATCH",
        CL_IMAGE_FORMAT_NOT_SUPPORTED : "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        CL_BUILD_PROGRAM_FAILURE : "CL_BUILD_PROGRAM_FAILURE",
        CL_MAP_FAILURE : "CL_MAP_FAILURE",
        CL_INVALID_VALUE : "CL_INVALID_VALUE",
        CL_INVALID_DEVICE_TYPE : "CL_INVALID_DEVICE_TYPE",
        CL_INVALID_PLATFORM : "CL_INVALID_PLATFORM",
        CL_INVALID_DEVICE : "CL_INVALID_DEVICE",
        CL_INVALID_CONTEXT : "CL_INVALID_CONTEXT",
        CL_INVALID_QUEUE_PROPERTIES : "CL_INVALID_QUEUE_PROPERTIES",
        CL_INVALID_COMMAND_QUEUE : "CL_INVALID_COMMAND_QUEUE",
        CL_INVALID_HOST_PTR : "CL_INVALID_HOST_PTR",
        CL_INVALID_MEM_OBJECT : "CL_INVALID_MEM_OBJECT",
        CL_INVALID_IMAGE_FORMAT_DESCRIPTOR : "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        CL_INVALID_IMAGE_SIZE : "CL_INVALID_IMAGE_SIZE",
        CL_INVALID_SAMPLER : "CL_INVALID_SAMPLER",
        CL_INVALID_BINARY : "CL_INVALID_BINARY",
        CL_INVALID_BUILD_OPTIONS : "CL_INVALID_BUILD_OPTIONS",
        CL_INVALID_PROGRAM : "CL_INVALID_PROGRAM",
        CL_INVALID_PROGRAM_EXECUTABLE : "CL_INVALID_PROGRAM_EXECUTABLE",
        CL_INVALID_KERNEL_NAME : "CL_INVALID_KERNEL_NAME",
        CL_INVALID_KERNEL_DEFINITION : "CL_INVALID_KERNEL_DEFINITION",
        CL_INVALID_KERNEL : "CL_INVALID_KERNEL",
        CL_INVALID_ARG_INDEX : "CL_INVALID_ARG_INDEX",
        CL_INVALID_ARG_VALUE : "CL_INVALID_ARG_VALUE",
        CL_INVALID_ARG_SIZE : "CL_INVALID_ARG_SIZE",
        CL_INVALID_KERNEL_ARGS : "CL_INVALID_KERNEL_ARGS",
        CL_INVALID_WORK_DIMENSION : "CL_INVALID_WORK_DIMENSION",
        CL_INVALID_WORK_GROUP_SIZE : "CL_INVALID_WORK_GROUP_SIZE",
        CL_INVALID_WORK_ITEM_SIZE : "CL_INVALID_WORK_ITEM_SIZE",
        CL_INVALID_GLOBAL_OFFSET : "CL_INVALID_GLOBAL_OFFSET",
        CL_INVALID_EVENT_WAIT_LIST : "CL_INVALID_EVENT_WAIT_LIST",
        CL_INVALID_EVENT : "CL_INVALID_EVENT",
        CL_INVALID_OPERATION : "CL_INVALID_OPERATION",
        CL_INVALID_GL_OBJECT : "CL_INVALID_GL_OBJECT",
        CL_INVALID_BUFFER_SIZE : "CL_INVALID_BUFFER_SIZE",
        CL_INVALID_MIP_LEVEL : "CL_INVALID_MIP_LEVEL",
        CL_INVALID_GLOBAL_WORK_SIZE : "CL_INVALID_GLOBAL_WORK_SIZE",
}


#
#
#   Args Translation
#
#

cdef union param:
    cl_mem          mem_value
    cl_sampler      sampler_value
    np.npy_byte         byte_value
    np.npy_ubyte         ubyte_value
    np.npy_short         short_value
    np.npy_ushort         ushort_value
    np.npy_int32         int32_value
    np.npy_uint32         uint32_value
    np.npy_int64         int64_value
    np.npy_uint64         uint64_value
    np.npy_intp         intp_value
    np.npy_float32         float32_value
    np.npy_float64         float64_value

ctypedef param (*param_converter_fct)(object) except *

cdef struct ptype:
    size_t itemsize
    param_converter_fct fct

cdef ptype param_converter_array[11 + 2]

cdef param from_byte(object val) except *:
    cdef param p
    p.byte_value = <np.npy_byte>val
    return p
param_converter_array[0].itemsize = sizeof(np.npy_byte)
param_converter_array[0].fct = from_byte

cdef param from_ubyte(object val) except *:
    cdef param p
    p.ubyte_value = <np.npy_ubyte>val
    return p
param_converter_array[1].itemsize = sizeof(np.npy_ubyte)
param_converter_array[1].fct = from_ubyte

cdef param from_short(object val) except *:
    cdef param p
    p.short_value = <np.npy_short>val
    return p
param_converter_array[2].itemsize = sizeof(np.npy_short)
param_converter_array[2].fct = from_short

cdef param from_ushort(object val) except *:
    cdef param p
    p.ushort_value = <np.npy_ushort>val
    return p
param_converter_array[3].itemsize = sizeof(np.npy_ushort)
param_converter_array[3].fct = from_ushort

cdef param from_int32(object val) except *:
    cdef param p
    p.int32_value = <np.npy_int32>val
    return p
param_converter_array[4].itemsize = sizeof(np.npy_int32)
param_converter_array[4].fct = from_int32

cdef param from_uint32(object val) except *:
    cdef param p
    p.uint32_value = <np.npy_uint32>val
    return p
param_converter_array[5].itemsize = sizeof(np.npy_uint32)
param_converter_array[5].fct = from_uint32

cdef param from_int64(object val) except *:
    cdef param p
    p.int64_value = <np.npy_int64>val
    return p
param_converter_array[6].itemsize = sizeof(np.npy_int64)
param_converter_array[6].fct = from_int64

cdef param from_uint64(object val) except *:
    cdef param p
    p.uint64_value = <np.npy_uint64>val
    return p
param_converter_array[7].itemsize = sizeof(np.npy_uint64)
param_converter_array[7].fct = from_uint64

cdef param from_intp(object val) except *:
    cdef param p
    p.intp_value = <np.npy_intp>val
    return p
param_converter_array[8].itemsize = sizeof(np.npy_intp)
param_converter_array[8].fct = from_intp

cdef param from_float32(object val) except *:
    cdef param p
    p.float32_value = <np.npy_float32>val
    return p
param_converter_array[9].itemsize = sizeof(np.npy_float32)
param_converter_array[9].fct = from_float32

cdef param from_float64(object val) except *:
    cdef param p
    p.float64_value = <np.npy_float64>val
    return p
param_converter_array[10].itemsize = sizeof(np.npy_float64)
param_converter_array[10].fct = from_float64

cdef param from_CLBuffer(object val) except *:
    cdef CLBuffer buf_val = val
    cdef param p
    p.mem_value = buf_val._mem
    return p
param_converter_array[11].itemsize = sizeof(cl_mem)
param_converter_array[11].fct = from_CLBuffer

cdef param from_CLSampler(object val) except *:
    cdef CLSampler buf_val = val
    cdef param p
    p.sampler_value = buf_val._sampler
    return p
param_converter_array[12].itemsize = sizeof(cl_sampler)
param_converter_array[12].fct = from_CLSampler


#
#
#   Helper functions
#
#



cdef size_t _getDeviceInfo_size_t(cl_device_id obj, cl_device_info param_name):
    cdef size_t size
    cdef size_t result
    cdef cl_int errcode = clGetDeviceInfo(obj, param_name, sizeof(size_t), &result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return result

cdef cl_ulong _getDeviceInfo_cl_ulong(cl_device_id obj, cl_device_info param_name):
    cdef size_t size
    cdef cl_ulong result
    cdef cl_int errcode = clGetDeviceInfo(obj, param_name, sizeof(cl_ulong), &result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return result

cdef bytes _getDeviceInfo_bytes(cl_device_id obj, cl_device_info param_name):
    cdef size_t size
    cdef char result[256]
    cdef cl_int errcode = clGetDeviceInfo(obj, param_name, 256 * sizeof(char), result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef bytes s = result[:size -1]
    return s

cdef cl_bool _getDeviceInfo_cl_bool(cl_device_id obj, cl_device_info param_name):
    cdef size_t size
    cdef cl_bool result
    cdef cl_int errcode = clGetDeviceInfo(obj, param_name, sizeof(cl_bool), &result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return result

cdef cl_uint _getDeviceInfo_cl_uint(cl_device_id obj, cl_device_info param_name):
    cdef size_t size
    cdef cl_uint result
    cdef cl_int errcode = clGetDeviceInfo(obj, param_name, sizeof(cl_uint), &result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return result



cdef bytes _getPlatformInfo_bytes(cl_platform_id obj, cl_platform_info param_name):
    cdef size_t size
    cdef char result[256]
    cdef cl_int errcode = clGetPlatformInfo(obj, param_name, 256 * sizeof(char), result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef bytes s = result[:size -1]
    return s



cdef size_t _getBufferInfo_size_t(cl_mem obj, cl_mem_info param_name):
    cdef size_t size
    cdef size_t result
    cdef cl_int errcode = clGetMemObjectInfo(obj, param_name, sizeof(size_t), &result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return result



cdef size_t _getImageInfo_size_t(cl_mem obj, cl_image_info param_name):
    cdef size_t size
    cdef size_t result
    cdef cl_int errcode = clGetImageInfo(obj, param_name, sizeof(size_t), &result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return result



cdef bytes _getKernelInfo_bytes(cl_kernel obj, cl_kernel_info param_name):
    cdef size_t size
    cdef char result[256]
    cdef cl_int errcode = clGetKernelInfo(obj, param_name, 256 * sizeof(char), result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef bytes s = result[:size -1]
    return s

cdef cl_uint _getKernelInfo_cl_uint(cl_kernel obj, cl_kernel_info param_name):
    cdef size_t size
    cdef cl_uint result
    cdef cl_int errcode = clGetKernelInfo(obj, param_name, sizeof(cl_uint), &result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return result



cdef cl_int _getEventInfo_cl_int(cl_event obj, cl_event_info param_name):
    cdef size_t size
    cdef cl_int result
    cdef cl_int errcode = clGetEventInfo(obj, param_name, sizeof(cl_int), &result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return result



cdef cl_ulong _getEventProfilingInfo_cl_ulong(cl_event obj, cl_profiling_info param_name):
    cdef size_t size
    cdef cl_ulong result
    cdef cl_int errcode = clGetEventProfilingInfo(obj, param_name, sizeof(cl_ulong), &result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return result



cdef cl_uint _getSamplerInfo_cl_uint(cl_sampler obj, cl_sampler_info param_name):
    cdef size_t size
    cdef cl_uint result
    cdef cl_int errcode = clGetSamplerInfo(obj, param_name, sizeof(cl_uint), &result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return result



DEF MAX_DEVICES_NUMBER = 10
cdef list _getDevices(cl_platform_id platform, cl_device_type dtype):
    cdef cl_device_id devices[MAX_DEVICES_NUMBER]
    cdef cl_uint num_devices
    cdef cl_int errcode
    errcode = clGetDeviceIDs(platform, dtype, MAX_DEVICES_NUMBER, devices, &num_devices)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLDevice instance
    cdef int i
    cdef list pydevices = []
    for i in xrange(num_devices):
        instance = CLDevice.__new__(CLDevice)
        instance._device = devices[i]
        pydevices.append(instance)
    return pydevices

cdef CLImage _createImage2D(CLContext context, size_t width, size_t height, cl_channel_order order, cl_channel_type itype):
    cdef cl_image_format format = [order, itype]
    cdef cl_uint offset = 0
    cdef cl_int errcode
    cdef cl_mem mem = clCreateImage2D(context._context, CL_MEM_READ_WRITE, &format, width, height, 0, NULL, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLImage instance = CLImage.__new__(CLImage)
    instance._mem = mem
    instance._context = context
    instance._offset = offset
    return instance

cdef CLImage _createImage3D(CLContext context, size_t width, size_t height, size_t depth, cl_channel_order order, cl_channel_type itype):
    cdef cl_image_format format = [order, itype]
    cdef cl_uint offset = 0
    cdef cl_int errcode
    cdef cl_mem mem = clCreateImage3D(context._context, CL_MEM_READ_WRITE, &format, width, height, depth, 0, 0, NULL, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLImage instance = CLImage.__new__(CLImage)
    instance._mem = mem
    instance._context = context
    instance._offset = offset
    return instance

cdef CLBuffer _createBuffer(CLContext context, size_t size, cl_mem_flags flags):
    cdef cl_uint offset = 0
    cdef cl_int errcode
    cdef cl_mem mem = clCreateBuffer(context._context, flags, size, NULL, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLBuffer instance = CLBuffer.__new__(CLBuffer)
    instance._mem = mem
    instance._context = context
    instance._offset = offset
    return instance

cdef CLCommandQueue _createCommandQueue(CLContext context, CLDevice device, cl_command_queue_properties flags):
    cdef cl_int errcode
    cdef cl_command_queue command_queue = clCreateCommandQueue(context._context, device._device, flags, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLCommandQueue instance = CLCommandQueue.__new__(CLCommandQueue)
    instance._context = context
    instance._command_queue = command_queue
    return instance

cdef CLSampler _createSampler(CLContext context, cl_bool normalized, cl_addressing_mode amode, cl_filter_mode fmode):
    cdef cl_int errcode
    cdef cl_sampler sampler = clCreateSampler(context._context, normalized, amode, fmode, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLSampler instance = CLSampler.__new__(CLSampler)
    instance._context = context
    instance._sampler = sampler
    return instance

cdef CLProgram _createProgramWithSource(CLContext context, bytes pystring):
    cdef const_char_ptr strings[1]
    strings[0] = pystring
    cdef size_t sizes = len(pystring)
    cdef cl_int errcode
    cdef cl_program program = clCreateProgramWithSource(context._context, 1, strings, &sizes, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLProgram instance = CLProgram.__new__(CLProgram)
    instance._context = context
    instance._program = program
    return instance

cdef CLKernel _createKernel(CLProgram program, bytes string):
    cdef cl_int errcode
    cdef cl_kernel kernel = clCreateKernel(program._program, string, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLKernel instance = CLKernel.__new__(CLKernel)
    instance._program = program
    instance._kernel = kernel
    return instance

cdef bytes _getBuildLog(CLProgram program, CLDevice device):
    cdef char log[10000]
    cdef size_t size
    cdef cl_int errcode
    errcode = clGetProgramBuildInfo(program._program, device._device, CL_PROGRAM_BUILD_LOG, 10000, log, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    s = log[:size]
    return s

cdef list _createKernelsInProgram(CLProgram program):
    cdef cl_kernel kernels[20]
    cdef cl_uint num_kernels
    cdef cl_int errcode
    errcode = clCreateKernelsInProgram(program._program, 20, kernels, &num_kernels)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef list pykernels = []
    cdef CLKernel instance
    cdef int i
    for i in xrange(num_kernels):
        instance = CLKernel.__new__(CLKernel)
        instance._kernel = kernels[i]
        instance._program = program
        pykernels.append(instance)
    return pykernels

cdef void _setArgs(CLKernel kernel, tuple args) except *:
    if len(args) != len(kernel._targs):
        raise AttributeError("Error")
    cdef int i
    cdef unsigned int index
    cdef param p
    cdef errcode
    for i in xrange(len(args)):
        index = kernel._targs[i]
        p = param_converter_array[index].fct(args[i])
        errcode = clSetKernelArg(kernel._kernel, i,param_converter_array[index].itemsize, &p)
        if errcode < 0: raise CLError(error_translation_table[errcode])

cdef void _setParameters(CLKernel kernel, tuple parameters) except *:
    cdef int i
    cdef unsigned int index
    #cdef unsigned int num_args = len(value)
    cdef cl_uint num_args = _getKernelInfo_cl_uint(kernel._kernel, CL_KERNEL_NUM_ARGS)
    if num_args != len(parameters):
        raise AttributeError("Number of args differ. got %d, expect %d" %                            (len(parameters), num_args))
    for i in xrange(num_args):
        index = parameters[i]
        if index >= 13:
            raise AttributeError("Unknown Type")


cdef void _enqueueWaitForEvents(cl_command_queue queue, list events) except *:
    cdef cl_event lst[100]
    cdef CLEvent evt
    cdef int i, num_events = min(100, len(events))
    for i from 0 <= i < num_events:
        evt = events[i]
        lst[i] = evt._event
    cdef cl_int errcode
    errcode = clEnqueueWaitForEvents(queue, num_events, lst)
    if errcode < 0: raise CLError(error_translation_table[errcode])

cdef CLEvent _enqueueNDRange(cl_command_queue queue, cl_kernel kernel, size_t gws[3], size_t lws[3]):
    cdef cl_event event
    cdef cl_int errcode
    errcode = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, gws, lws, 0, NULL, &event)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLEvent instance = CLEvent.__new__(CLEvent)
    instance._event = event
    return instance

cdef CLEvent _enqueueWriteBuffer(cl_command_queue queue, cl_mem buffer,
                                 cl_bool blocking, size_t offset, size_t cb,
                                 void *ptr):
    cdef cl_event event
    cdef cl_int errcode
    errcode = clEnqueueWriteBuffer(queue, buffer, blocking, offset, cb, ptr, 0, NULL, &event)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLEvent instance = CLEvent.__new__(CLEvent)
    instance._event = event
    return instance

cdef CLEvent _enqueueReadBuffer(cl_command_queue queue, cl_mem buffer,
                                 cl_bool blocking, size_t offset, size_t cb,
                                 void *ptr):
    cdef cl_event event
    cdef cl_int errcode
    errcode = clEnqueueReadBuffer(queue, buffer, blocking, offset, cb, ptr, 0, NULL, &event)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLEvent instance = CLEvent.__new__(CLEvent)
    instance._event = event
    return instance

cdef CLEvent _enqueueCopyBuffer(cl_command_queue queue, cl_mem src,cl_mem dst,
                                size_t src_offset, size_t dst_offset, size_t size, cl_bool blocking):
    cdef cl_event event
    cdef cl_int errcode
    errcode = clEnqueueCopyBuffer(queue, src, dst, src_offset, dst_offset, size, 0, NULL, &event)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLEvent instance = CLEvent.__new__(CLEvent)
    instance._event = event
    return instance

cdef CLEvent _enqueueReadImage (cl_command_queue queue, cl_mem image, cl_bool blocking,
                                size_t origin[3], size_t region[3], size_t row_pitch, size_t slice_pitch, void *ptr):
    cdef cl_event event
    cdef cl_int errcode
    errcode = clEnqueueReadImage(queue, image, blocking, origin, region, row_pitch, slice_pitch, ptr, 0, NULL, &event)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLEvent instance = CLEvent.__new__(CLEvent)
    instance._event = event
    return instance

cdef CLEvent _enqueueWriteImage (cl_command_queue queue, cl_mem image, cl_bool blocking,
                                 size_t origin[3], size_t region[3], size_t row_pitch, size_t slice_pitch, void *ptr):
    cdef cl_event event
    cdef cl_int errcode
    errcode = clEnqueueWriteImage(queue, image, blocking, origin, region, row_pitch, slice_pitch, ptr, 0, NULL, &event)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLEvent instance = CLEvent.__new__(CLEvent)
    instance._event = event
    return instance

cdef CLEvent _enqueueMapBuffer(cl_command_queue queue, cl_mem src, cl_bool blocking,
                               cl_map_flags flags, size_t offset, size_t size, void **dst):
    cdef cl_event event
    cdef cl_int errcode
    dst[0] = clEnqueueMapBuffer(queue, src, blocking, flags, offset, size, 0, NULL, &event, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLEvent instance = CLEvent.__new__(CLEvent)
    instance._event = event
    return instance

cdef CLEvent _enqueueUnmapMemObject(cl_command_queue queue, cl_mem mem, void *ptr):
    cdef cl_event event
    cdef cl_int errcode
    errcode = clEnqueueUnmapMemObject(queue, mem, ptr, 0, NULL, &event)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLEvent instance = CLEvent.__new__(CLEvent)
    instance._event = event
    return instance

cdef void _build(CLProgram program, list options):
    cdef cl_int errcode
    errcode = clBuildProgram(program._program, 0, NULL, NULL, NULL, NULL)
    if errcode < 0: raise CLError(error_translation_table[errcode])


#
#
#   Classes
#
#

cdef class CLCommandQueue(CLObject):
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseCommandQueue(self._command_queue) 
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % self.__class__.__name__)

    def flush(self):
        cdef cl_int errcode
        errcode = clFlush(self._command_queue)
        if errcode < 0: raise CLError(error_translation_table[errcode])

    def finish(self):
        cdef cl_int errcode
        errcode = clFinish(self._command_queue)
        if errcode < 0: raise CLError(error_translation_table[errcode])

    def enqueueWriteBuffer(self, CLBuffer mem, np.ndarray ary, bint blocking = True):
        cdef unsigned int copy_size = ary.size * ary.dtype.itemsize
        if (mem.size - mem_offset) != copy.size:
            raise AttributeError("Size Mismatch")
        return _enqueueWriteBuffer(self._command_queue, mem._mem, blocking, mem._offset, copy_size, ary.data)

    def enqueueReadBuffer(self, np.ndarray ary, CLBuffer mem, bint blocking = True):
        cdef unsigned int copy_size = mem.size - mem._offset
        if copy_size  != ary.size * ary.dtype.itemsize:
            raise AttributeError("Size mismatch")
        return _enqueueReadBuffer(self._command_queue, mem._mem, blocking, mem._offset, copy_size, ary.data)

    def enqueueCopyBuffer(self, CLBuffer src, CLBuffer dst, bint blocking = True):
        cdef unsigned int copy_size = src.size - src._offset
        if copy_size != dst.size - dst._offset:
            raise AttributeError("Size mismatch")
        return _enqueueCopyBuffer(self._command_queue, src._mem, dst._mem, src._offset, dst._offset, copy_size, blocking)

    def enqueueUnmapMemObject(self, CLMappedBuffer buffer):
        cdef CLEvent event = _enqueueUnmapMemObject(self._command_queue, buffer._buffer._mem, buffer._address)
        buffer._mapped = False
        return event

    def enqueueBarrier(self): 
        cdef cl_int errcode
        errcode = clEnqueueBarrier(self._command_queue)
        if errcode < 0: raise CLError(error_translation_table[errcode])

    def enqueueMarker(self): 
        cdef cl_event event
        cdef cl_int errcode
        errcode = clEnqueueMarker(self._command_queue, &event)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        cdef CLEvent instance = CLEvent.__new__(CLEvent)
        instance._event = event
        return instance

    def enqueueWaitForEvents(self, list events): _enqueueWaitForEvents(self._command_queue, events)

    def enqueueReadImage(self, np.ndarray ary, CLImage mem, bint blocking = True):
        cdef size_t shape[3], origin[3], pitch = 0, slice = 0
        shape[1] = shape[2] = origin[0] = origin[1] = origin[2] = 0
        shape[0] = ary.shape[0]
        if ary.ndim > 1:
            shape[1] = ary.shape[1]
            pitch = ary.strides[0]
        if ary.ndim > 2:
            shape[2] = ary.shape[2]
            pitch = ary.strides[1]
            slice = ary.strides[0]
        return _enqueueReadImage(self._command_queue, mem._mem, blocking, origin, shape, pitch, slice, ary.data)

    def enqueueWriteImage(self, np.ndarray ary, CLImage mem, bint blocking = True):
        cdef size_t shape[3], origin[3], pitch = 0, slice = 0
        shape[0] = shape[1] = shape[2] = origin[0] = origin[1] = origin[2] = 0
        if ary.ndim > 1:
            shape[1] = ary.shape[1]
            pitch = ary.strides[0]
        if ary.ndim > 2:
            shape[2] = ary.shape[2]
            pitch = ary.strides[1]
            slice = ary.strides[0]
        return _enqueueWriteImage(self._command_queue, mem._mem, blocking, origin, shape, pitch, slice, ary.data)

    def enqueueNDRange(self, CLKernel kernel, tuple global_work_size = (1,1,1), tuple local_work_size = (1,1,1)):
        cdef size_t gws[3]
        gws[0] = global_work_size[0]
        gws[1] = global_work_size[1]
        gws[2] = global_work_size[2]
        cdef size_t lws[3]
        lws[0] = local_work_size[0]
        lws[1] = local_work_size[1]
        lws[2] = local_work_size[2]
        return _enqueueNDRange(self._command_queue, kernel._kernel, gws, lws)

    def enqueueMapBuffer(self, CLBuffer buffer, cl_map_flags flags = CL_MAP_WRITE | CL_MAP_READ, bint blocking = True):
        cdef size_t copy_size = buffer.size - buffer._offset
        cdef void *address
        cdef CLEvent event = _enqueueMapBuffer(self._command_queue, buffer._mem, blocking,flags, buffer._offset, copy_size, &address)
        cdef CLMappedBuffer instance = CLMappedBuffer.__new__(CLMappedBuffer)
        instance._address = address
        instance._buffer = buffer

        instance._mapped = True
        instance._command_queue = self
        return event, instance


cdef class CLProgram(CLObject):
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseProgram(self._program) 
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % self.__class__.__name__)

    def createKernelsInProgram(self): return _createKernelsInProgram(self)

    def createKernel(self, bytes string): return _createKernel(self, string)

    def getBuildLog(self, CLDevice device): return _getBuildLog(self, device)

    def build(self, list options = []):
        _build(self, options)
        return self

cdef class CLMappedBuffer(CLObject):

    def __repr__(self):
        return '<%s address="%s" size="%s">' % (self.__class__.__name__, self.address, self.size, )

    property address:
        def __get__(self):
            return <np.Py_intptr_t> self._address
    property __array_interface__:
        def __get__(self):
            return { "shape" : (self._buffer.size,),
                     "typestr" : "|i1",
                     "data" : (<np.Py_intptr_t> self._address, False),
                     "version" : 3}
    property size:
        def __get__(self):
            return self._buffer.size

    def __dealloc__(self):
        if self._mapped:
            self._command_queue.enqueueUnmapMemObject(self)


cdef class CLDevice(CLObject):

    property maxWorkGroupSize:
        def __get__(self):
            return _getDeviceInfo_size_t(self._device,
                                        CL_DEVICE_MAX_WORK_GROUP_SIZE)
    property profilingTimerResolution:
        def __get__(self):
            return _getDeviceInfo_size_t(self._device,
                                        CL_DEVICE_PROFILING_TIMER_RESOLUTION)
    property image2DMaxSize:
        def __get__(self):
                cdef size_t r_0 = _getDeviceInfo_size_t(self._device,
                                        CL_DEVICE_IMAGE2D_MAX_HEIGHT)
                cdef size_t r_1 = _getDeviceInfo_size_t(self._device,
                                        CL_DEVICE_IMAGE2D_MAX_WIDTH)
                return (r_0, r_1, )

    property image3DMaxSize:
        def __get__(self):
                cdef size_t r_0 = _getDeviceInfo_size_t(self._device,
                                        CL_DEVICE_IMAGE3D_MAX_HEIGHT)
                cdef size_t r_1 = _getDeviceInfo_size_t(self._device,
                                        CL_DEVICE_IMAGE3D_MAX_WIDTH)
                cdef size_t r_2 = _getDeviceInfo_size_t(self._device,
                                        CL_DEVICE_IMAGE3D_MAX_DEPTH)
                return (r_0, r_1, r_2, )

    property globalMemSize:
        def __get__(self):
            return _getDeviceInfo_cl_ulong(self._device,
                                        CL_DEVICE_GLOBAL_MEM_SIZE)
    property globalMemCacheSize:
        def __get__(self):
            return _getDeviceInfo_cl_ulong(self._device,
                                        CL_DEVICE_GLOBAL_MEM_CACHE_SIZE)
    property globalMemCachelineSize:
        def __get__(self):
            return _getDeviceInfo_cl_ulong(self._device,
                                        CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE)
    property maxConstantBufferSize:
        def __get__(self):
            return _getDeviceInfo_cl_ulong(self._device,
                                        CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE)
    property maxMemAllocSize:
        def __get__(self):
            return _getDeviceInfo_cl_ulong(self._device,
                                        CL_DEVICE_MAX_MEM_ALLOC_SIZE)
    property type:
        def __get__(self):
            return _getDeviceInfo_cl_ulong(self._device,
                                        CL_DEVICE_TYPE)
    property driverVersion:
        def __get__(self):
            return _getDeviceInfo_bytes(self._device,
                                        CL_DRIVER_VERSION)
    property vendor:
        def __get__(self):
            return _getDeviceInfo_bytes(self._device,
                                        CL_DEVICE_VERSION)
    property version:
        def __get__(self):
            return _getDeviceInfo_bytes(self._device,
                                        CL_DEVICE_VENDOR)
    property profile:
        def __get__(self):
            return _getDeviceInfo_bytes(self._device,
                                        CL_DRIVER_PROFILE)
    property name:
        def __get__(self):
            return _getDeviceInfo_bytes(self._device,
                                        CL_DEVICE_NAME)
    property extensions:
        def __get__(self):
            return _getDeviceInfo_bytes(self._device,
                                        CL_DEVICE_EXTENSIONS)
    property imageSupport:
        def __get__(self):
            return _getDeviceInfo_cl_bool(self._device,
                                        CL_DEVICE_IMAGE_SUPPORT)
    property ECCSupport:
        def __get__(self):
            return _getDeviceInfo_cl_bool(self._device,
                                        CL_DEVICE_ERROR_CORRECTION_SUPPORT)
    property endianLittle:
        def __get__(self):
            return _getDeviceInfo_cl_bool(self._device,
                                        CL_DEVICE_ENDIAN_LITTLE)
    property compilerAvailable:
        def __get__(self):
            return _getDeviceInfo_cl_bool(self._device,
                                        CL_DEVICE_COMPILER_AVAILABLE)
    property available:
        def __get__(self):
            return _getDeviceInfo_cl_bool(self._device,
                                        CL_DEVICE_AVAILABLE)
    property addressBits:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_ADDRESS_BITS)
    property vendorId:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_VENDOR_ID)
    property maxClockFrequency:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_MAX_CLOCK_FREQUENCY)
    property maxComputeUnits:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_MAX_COMPUTE_UNITS)
    property maxWorkItemDimensions:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)
    property maxConstantArgs:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_MAX_CONSTANT_ARGS)
    property minDataTypeAlignSize:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE)
    property maxWriteImageArgs:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_MAX_WRITE_IMAGE_ARGS)
    property memBaseAddrAlign:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_MEM_BASE_ADDR_ALIGN)
    property preferredVectorWidthChar:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR)
    property preferredVectorWidthShort:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT)
    property preferredVectorWidthInt:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT)
    property preferredVectorWidthLong:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG)
    property preferredVectorWidthFloat:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT)
    property preferredVectorWidthDouble:
        def __get__(self):
            return _getDeviceInfo_cl_uint(self._device,
                                        CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE)


    def __repr__(self):
        return '<%s name="%s" type="%s" vendor="%s" driverVersion="%s">' % (self.__class__.__name__, self.name, self.type, self.vendor, self.driverVersion, )


cdef class CLPlatform(CLObject):

    property version:
        def __get__(self):
            return _getPlatformInfo_bytes(self._platform,
                                        CL_PLATFORM_VERSION)
    property name:
        def __get__(self):
            return _getPlatformInfo_bytes(self._platform,
                                        CL_PLATFORM_NAME)
    property vendor:
        def __get__(self):
            return _getPlatformInfo_bytes(self._platform,
                                        CL_PLATFORM_VENDOR)
    property extensions:
        def __get__(self):
            return _getPlatformInfo_bytes(self._platform,
                                        CL_PLATFORM_EXTENSIONS)
    property profile:
        def __get__(self):
            return _getPlatformInfo_bytes(self._platform,
                                        CL_PLATFORM_PROFILE)


    def __repr__(self):
        return '<%s name="%s" vendor="%s" version="%s">' % (self.__class__.__name__, self.name, self.vendor, self.version, )

    def getDevices(self, cl_device_type dtype = 0xFFFFFFFF): return _getDevices(self._platform, dtype)

    def build(self, list options = []): _build(self, options)

cdef class CLBuffer(CLObject):
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseMemObject(self._mem) 
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % self.__class__.__name__)


    property size:
        def __get__(self):
            return _getBufferInfo_size_t(self._mem,
                                        CL_MEM_SIZE)


    def __repr__(self):
        return '<%s size="%s" offset="%s">' % (self.__class__.__name__, self.size, self.offset, )

    property offset:
        def __get__(self):
            return self._offset


cdef class CLImage(CLBuffer):

    property slicePitch:
        def __get__(self):
            return _getImageInfo_size_t(self._mem,
                                        CL_IMAGE_SLICE_PITCH)
    property elementSize:
        def __get__(self):
            return _getImageInfo_size_t(self._mem,
                                        CL_IMAGE_ELEMENT_SIZE)
    property shape:
        def __get__(self):
                cdef size_t r_0 = _getImageInfo_size_t(self._mem,
                                        CL_IMAGE_WIDTH)
                cdef size_t r_1 = _getImageInfo_size_t(self._mem,
                                        CL_IMAGE_HEIGHT)
                cdef size_t r_2 = _getImageInfo_size_t(self._mem,
                                        CL_IMAGE_DEPTH)
                return (r_0, r_1, r_2, )

    property rowPitch:
        def __get__(self):
            return _getImageInfo_size_t(self._mem,
                                        CL_IMAGE_ROW_PITCH)


    def __repr__(self):
        return '<%s shape="%s">' % (self.__class__.__name__, self.shape, )



cdef class CLKernel(CLObject):

    property name:
        def __get__(self):
            return _getKernelInfo_bytes(self._kernel,
                                        CL_KERNEL_FUNCTION_NAME)
    property numArgs:
        def __get__(self):
            return _getKernelInfo_cl_uint(self._kernel,
                                        CL_KERNEL_NUM_ARGS)


    def __repr__(self):
        return '<%s name="%s" numArgs="%s">' % (self.__class__.__name__, self.name, self.numArgs, )

    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseKernel(self._kernel) 
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % self.__class__.__name__)

    property parameters:
        def __get__(self):
            return self._targs

        def __set__(self, tuple value):
            _setParameters(self, value)
            self._targs = value

    def setArgs(self, *args): _setArgs(self, args)


cdef class CLEvent(CLObject):
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseEvent(self._event) 
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % self.__class__.__name__)


    property type:
        def __get__(self):
            return _getEventInfo_cl_int(self._event,
                                        CL_EVENT_COMMAND_TYPE)
    property status:
        def __get__(self):
            return _getEventInfo_cl_int(self._event,
                                        CL_EVENT_COMMAND_EXECUTION_STATUS)


    property profilingQueued:
        def __get__(self):
            return _getEventProfilingInfo_cl_ulong(self._event,
                                        CL_PROFILING_COMMAND_QUEUED)
    property profilingSubmit:
        def __get__(self):
            return _getEventProfilingInfo_cl_ulong(self._event,
                                        CL_PROFILING_COMMAND_SUBMIT)
    property profilingStart:
        def __get__(self):
            return _getEventProfilingInfo_cl_ulong(self._event,
                                        CL_PROFILING_COMMAND_START)
    property profilingEnd:
        def __get__(self):
            return _getEventProfilingInfo_cl_ulong(self._event,
                                        CL_PROFILING_COMMAND_END)



cdef class CLSampler(CLObject):
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseSampler(self._sampler) 
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % self.__class__.__name__)


    property normalized:
        def __get__(self):
            return _getSamplerInfo_cl_uint(self._sampler,
                                        CL_SAMPLER_NORMALIZED_COORDS)
    property filterMode:
        def __get__(self):
            return _getSamplerInfo_cl_uint(self._sampler,
                                        CL_SAMPLER_FILTER_MODE)
    property addressingMode:
        def __get__(self):
            return _getSamplerInfo_cl_uint(self._sampler,
                                        CL_SAMPLER_ADDRESSING_MODE)


    def __repr__(self):
        return '<%s normalized="%s" filterMode="%s" addressingMode="%s">' % (self.__class__.__name__, self.normalized, self.filterMode, self.addressingMode, )


cdef class CLContext(CLObject):
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = clReleaseContext (self._context) 
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % self.__class__.__name__)


    def __repr__(self):
        return '<%s>' % (self.__class__.__name__, )

    def createBuffer(self, size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE):
        return _createBuffer(self, size, flags)
    def createImage2D(self, size_t width, size_t height, cl_channel_order order, cl_channel_type itype):
        return _createImage2D(self, width, height, order, itype)
    def createImage3D(self, size_t width, size_t height, size_t depth, cl_channel_order order, cl_channel_type itype):
        return _createImage3D(self, width, height, depth, order, itype)
    def createCommandQueue(self, CLDevice device, cl_command_queue_properties flags = <cl_command_queue_properties>0):
        return _createCommandQueue(self, device, flags)
    def createSampler(self, cl_bool normalized, cl_addressing_mode amode, cl_filter_mode fmode):
        return _createSampler(self, normalized, amode, fmode)
    def createProgramWithSource(self, bytes pystring):
        return _createProgramWithSource(self, pystring)

    property devices:
        def __get__(self):
            return self._devices



#
#
#   Module level API
#
#

DEF MAX_PLATFORM_NUMBER = 15
cpdef list getPlatforms():
    cdef list pyplatforms = []
    cdef cl_platform_id platforms[MAX_PLATFORM_NUMBER]
    cdef cl_uint num_platforms
    cdef CLPlatform instance
    cdef int i
    cdef cl_int errcode
    errcode = clGetPlatformIDs(MAX_PLATFORM_NUMBER, platforms, &num_platforms)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    for i in xrange(num_platforms):
        instance = CLPlatform.__new__(CLPlatform)
        instance._platform = platforms[i]
        pyplatforms.append(instance)
    return pyplatforms

cpdef CLContext createContext(list devices):
    cdef long num_devices = len(devices)
    cdef cl_device_id clDevices[100]
    cdef cl_context context
    for i from 0 <= i < min(num_devices, 100):
        clDevices[i] = (<CLDevice>devices[i])._device
    cdef cl_int errcode
    context = clCreateContext(NULL, num_devices, clDevices, NULL, NULL, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLContext instance = CLContext.__new__(CLContext)
    instance._context = context
    instance._devices = devices
    return instance

cpdef waitForEvents(list events):
    cdef int num_events = len(events)
    cdef cl_event lst[100]
    cdef CLEvent evt

    for i from 0 <= i < min(num_events, 100):
        lst[i] = (<CLEvent>events[i])._event
    cdef cl_int errcode
    errcode = clWaitForEvents(num_events, lst)
    if errcode < 0: raise CLError(error_translation_table[errcode])
















