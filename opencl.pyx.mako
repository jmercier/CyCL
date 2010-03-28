<%
from itertools import izip
param_types = ['byte', 'ubyte',
               'short', 'ushort',
               'int32', 'uint32',
               'int64', 'uint64',
               'intp',
               'float32',
               'float64',
               'float128'];


error_types = ['CL_SUCCESS',
               'CL_DEVICE_NOT_FOUND',
               'CL_DEVICE_NOT_AVAILABLE',
               'CL_COMPILER_NOT_AVAILABLE',
               'CL_MEM_OBJECT_ALLOCATION_FAILURE',
               'CL_OUT_OF_RESOURCES',
               'CL_OUT_OF_HOST_MEMORY',
               'CL_PROFILING_INFO_NOT_AVAILABLE',
               'CL_MEM_COPY_OVERLAP',
               'CL_IMAGE_FORMAT_MISMATCH',
               'CL_IMAGE_FORMAT_NOT_SUPPORTED',
               'CL_BUILD_PROGRAM_FAILURE',
               'CL_MAP_FAILURE',
               'CL_INVALID_VALUE',
               'CL_INVALID_DEVICE_TYPE',
               'CL_INVALID_PLATFORM',
               'CL_INVALID_DEVICE',
               'CL_INVALID_CONTEXT',
               'CL_INVALID_QUEUE_PROPERTIES',
               'CL_INVALID_COMMAND_QUEUE',
               'CL_INVALID_HOST_PTR',
               'CL_INVALID_MEM_OBJECT',
               'CL_INVALID_IMAGE_FORMAT_DESCRIPTOR',
               'CL_INVALID_IMAGE_SIZE',
               'CL_INVALID_SAMPLER',
               'CL_INVALID_BINARY',
               'CL_INVALID_BUILD_OPTIONS',
               'CL_INVALID_PROGRAM',
               'CL_INVALID_PROGRAM_EXECUTABLE',
               'CL_INVALID_KERNEL_NAME',
               'CL_INVALID_KERNEL_DEFINITION',
               'CL_INVALID_KERNEL',
               'CL_INVALID_ARG_INDEX',
               'CL_INVALID_ARG_VALUE',
               'CL_INVALID_ARG_SIZE',
               'CL_INVALID_KERNEL_ARGS',
               'CL_INVALID_WORK_DIMENSION',
               'CL_INVALID_WORK_GROUP_SIZE',
               'CL_INVALID_WORK_ITEM_SIZE',
               'CL_INVALID_GLOBAL_OFFSET',
               'CL_INVALID_EVENT_WAIT_LIST',
               'CL_INVALID_EVENT',
               'CL_INVALID_OPERATION',
               'CL_INVALID_GL_OBJECT',
               'CL_INVALID_BUFFER_SIZE',
               'CL_INVALID_MIP_LEVEL',
               'CL_INVALID_GLOBAL_WORK_SIZE']

device_properties = { 'bytes'  :  [('driverVersion',            'CL_DRIVER_VERSION'),
                                   ('vendor',                   'CL_DEVICE_VERSION'),
                                   ('version',                  'CL_DEVICE_VENDOR',),
                                   ('profile',                  'CL_DRIVER_PROFILE'),
                                   ('name',                     'CL_DEVICE_NAME'),
                                   ('extensions',               'CL_DEVICE_EXTENSIONS')],
                     'cl_uint'  : [('addressBits',              'CL_DEVICE_ADDRESS_BITS'),
                                   ('vendorId',                 'CL_DEVICE_VENDOR_ID'),
                                   ('maxClockFrequency',        'CL_DEVICE_MAX_CLOCK_FREQUENCY'),
                                   ('maxComputeUnits',           'CL_DEVICE_MAX_COMPUTE_UNITS'),
                                   ('maxWorkItemDimensions',    'CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS'),
                                   ('maxConstantArgs',          'CL_DEVICE_MAX_CONSTANT_ARGS'),
                                   ('minDataTypeAlignSize',      'CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE'),
                                   ('maxWriteImageArgs',        'CL_DEVICE_MAX_WRITE_IMAGE_ARGS'),
                                   ('memBaseAddrAlign',         'CL_DEVICE_MEM_BASE_ADDR_ALIGN'),
                                   ('preferredVectorWidthChar', 'CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR'),
                                   ('preferredVectorWidthShort','CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT'),
                                   ('preferredVectorWidthInt',  'CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT'),
                                   #('maxWorkItemSizes',         'CL_DEVICE_MAX_WORK_ITEM_SIZES'),
                                   ('preferredVectorWidthLong', 'CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG'),
                                   ('preferredVectorWidthFloat','CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT'),
                                   ('preferredVectorWidthDouble','CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE')],
                     'size_t'   : [('maxWorkGroupSize',         'CL_DEVICE_MAX_WORK_GROUP_SIZE'),
                                   ('profilingTimerResolution', 'CL_DEVICE_PROFILING_TIMER_RESOLUTION'),
                                   ('image2DMaxSize',           ('CL_DEVICE_IMAGE2D_MAX_HEIGHT', 'CL_DEVICE_IMAGE2D_MAX_WIDTH')),
                                   ('image3DMaxSize',           ('CL_DEVICE_IMAGE3D_MAX_HEIGHT', 'CL_DEVICE_IMAGE3D_MAX_WIDTH','CL_DEVICE_IMAGE3D_MAX_DEPTH'))],
                     'cl_ulong' : [('globalMemSize',            'CL_DEVICE_GLOBAL_MEM_SIZE'),
                                   ('globalMemCacheSize',       'CL_DEVICE_GLOBAL_MEM_CACHE_SIZE'),
                                   ('globalMemCachelineSize',   'CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE'),
                                   ('maxConstantBufferSize',    'CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE'),
                                   ('maxMemAllocSize',          'CL_DEVICE_MAX_MEM_ALLOC_SIZE'),
                                   ('type',                     'CL_DEVICE_TYPE')],
                     'cl_bool'  : [('imageSupport',             'CL_DEVICE_IMAGE_SUPPORT'),
                                   ('ECCSupport',            'CL_DEVICE_ERROR_CORRECTION_SUPPORT'),
                                   ('endianLittle',          'CL_DEVICE_ENDIAN_LITTLE'),
                                   ('compilerAvailable',     'CL_DEVICE_COMPILER_AVAILABLE'),
                                   ('available',             'CL_DEVICE_AVAILABLE')]}


platform_properties = { 'bytes'  : [('version',             'CL_PLATFORM_VERSION'),
                                    ('name',                'CL_PLATFORM_NAME'),
                                    ('vendor',              'CL_PLATFORM_VENDOR'),
                                    ('extensions',          'CL_PLATFORM_EXTENSIONS'),
                                    ('profile',             'CL_PLATFORM_PROFILE')]}

buffer_properties = { 'size_t'   : [('size',                'CL_MEM_SIZE')]}

image_properties = { 'size_t'  : [('slicePitch',            'CL_IMAGE_SLICE_PITCH'),
                                  ('elementSize',           'CL_IMAGE_ELEMENT_SIZE'),
                                  ('shape',                 ('CL_IMAGE_WIDTH', 'CL_IMAGE_HEIGHT', 'CL_IMAGE_DEPTH')),
                                  ('rowPitch',              'CL_IMAGE_ROW_PITCH')]}

kernel_properties = { 'bytes'   : [('name', 'CL_KERNEL_FUNCTION_NAME')],
                      'cl_uint' : [('numArgs', 'CL_KERNEL_NUM_ARGS')]}

event_properties = { 'cl_int'     : [('type', 'CL_EVENT_COMMAND_TYPE'),
                                     ('status', 'CL_EVENT_COMMAND_EXECUTION_STATUS')]}

profiling_properties = { 'cl_ulong'   : [('profilingQueued', 'CL_PROFILING_COMMAND_QUEUED'),
                                               ('profilingSubmit', 'CL_PROFILING_COMMAND_SUBMIT'),
                                               ('profilingStart',  'CL_PROFILING_COMMAND_START'),
                                               ('profilingEnd',    'CL_PROFILING_COMMAND_END')]}

sampler_properties = { 'cl_uint'    : [('normalized', 'CL_SAMPLER_NORMALIZED_COORDS'),
                                       ('filterMode', 'CL_SAMPLER_FILTER_MODE'),
                                       ('addressingMode', 'CL_SAMPLER_ADDRESSING_MODE')]}

%>

${copyright()}
cimport opencl
cimport numpy as np
from opencl cimport *

from defines import *

cdef dict error_translation_table = {
%for e in error_types:
        ${e} : "${e}",
%endfor

}


${makesection("Args Translation")}
cdef union param:
%for t in param_types:
    np.npy_${t}         ${t}_value
%endfor

ctypedef param (*param_converter_fct)(object) except *

cdef struct ptype:
    size_t itemsize
    param_converter_fct fct

cdef ptype param_converter_array[${len(param_types)}]

%for i, t in enumerate(param_types):
cdef param from_${t}(object val) except *:
    cdef param p
    p.${t}_value = <np.npy_${t}>val
    return p
param_converter_array[${i}].itemsize = sizeof(np.npy_${t})
param_converter_array[${i}].fct = from_${t}

%endfor

${makesection("Helper functions")}

${info_getter("getDeviceInfo","clGetDeviceInfo", "cl_device_id", "cl_device_info", device_properties)}
${info_getter("getPlatformInfo","clGetPlatformInfo", "cl_platform_id", "cl_platform_info", platform_properties)}
${info_getter("getBufferInfo","clGetMemObjectInfo", "cl_mem", "cl_mem_info", buffer_properties)}
${info_getter("getImageInfo","clGetImageInfo", "cl_mem", "cl_image_info", image_properties)}
${info_getter("getKernelInfo","clGetKernelInfo", "cl_kernel", "cl_kernel_info", kernel_properties)}
${info_getter("getEventInfo","clGetEventInfo", "cl_event", "cl_event_info", event_properties)}
${info_getter("getEventProfilingInfo","clGetEventProfilingInfo", "cl_event", "cl_profiling_info", profiling_properties)}
${info_getter("getSamplerInfo","clGetSamplerInfo", "cl_sampler", "cl_sampler_info", sampler_properties)}

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
    cdef cl_image_format format
    cdef cl_mem mem
    cdef CLImage instance
    cdef cl_int errcode
    cdef cl_uint offset = 0

    format.image_channel_order = order
    format.image_channel_data_type = itype
    mem = clCreateImage2D(context._context, CL_MEM_READ_WRITE, &format, width, height, 0, NULL, &errcode)
    if errcode < 0: raise translateError(errcode)

    instance = CLImage.__new__(CLImage)\
${init_instance(['mem', 'context', 'offset'], '    ')}
    return instance

cdef CLImage _createImage3D(CLContext context, size_t width, size_t height, size_t depth, cl_channel_order order, cl_channel_type itype):
    cdef cl_int errcode
    cdef cl_image_format format
    cdef cl_uint offset = 0
    cdef cl_mem mem

    format.image_channel_order = order
    format.image_channel_data_type = itype
    mem = clCreateImage3D(context._context, CL_MEM_READ_WRITE, &format, width, height, depth, 0, 0, NULL, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLImage instance = CLImage.__new__(CLImage) \
${init_instance(['mem', 'context', 'offset'], '    ')}
    return instance

cdef CLBuffer _createBuffer(CLContext context, size_t size, cl_mem_flags flags):
    cdef cl_int errcode
    cdef cl_mem mem
    cdef cl_uint offset = 0
    mem = clCreateBuffer(context._context, flags, size, NULL, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLBuffer instance = CLBuffer.__new__(CLBuffer) \
${init_instance(['mem', 'context', 'offset'], '    ')}
    return instance

cdef CLCommandQueue _createCommandQueue(CLContext context, CLDevice device, cl_command_queue_properties flags):
    cdef cl_command_queue command_queue
    cdef cl_int errcode
    command_queue = clCreateCommandQueue(context._context, device._device, flags, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLCommandQueue instance = CLCommandQueue.__new__(CLCommandQueue)
${init_instance(['context', 'command_queue'], '    ')}
    return instance

cdef CLSampler _createSampler(CLContext context, cl_bool normalized, cl_addressing_mode amode, cl_filter_mode fmode):
    cdef cl_sampler sampler
    cdef cl_int errcode
    sampler = clCreateSampler(context._context, normalized, amode, fmode, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLSampler instance = CLSampler.__new__(CLSampler)
${init_instance(['context', 'sampler'], '    ')}
    return instance

cdef CLProgram _createProgramWithSource(CLContext context, bytes pystring):
    cdef cl_program program
    cdef const_char_ptr strings[1]
    cdef cl_int errcode
    strings[0] = pystring
    cdef size_t sizes = len(pystring)
    program = clCreateProgramWithSource(context._context, 1, strings, &sizes, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLProgram instance = CLProgram.__new__(CLProgram)
${init_instance(['context', 'program'], '    ')}
    return instance

cdef CLKernel _createKernel(CLProgram program, bytes string):
    cdef cl_kernel kernel
    cdef cl_int errcode
    kernel = clCreateKernel(program._program, string, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef CLKernel instance = CLKernel.__new__(CLKernel)
${init_instance(['program', 'kernel'], '    ')}
    return instance

cdef bytes getBuildLog(CLProgram program, CLDevice device):
    cdef char log[10000]
    cdef size_t size
    cdef cl_int errcode
    errcode = clGetProgramBuildInfo(program._program, device._device, CL_PROGRAM_BUILD_LOG, MAX_PROGRAM_BUILD_LOG, log, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    s = log[:size]
    return s

cdef list _createKernelsInProgram(CLProgram program):
    cdef cl_kernel kernels[20]
    cdef cl_int num_kernels
    cdef cl_int errcode
    errcode = clCreateKernelsInProgram(program._program, MAX_KERNELS_IN_PROGRAM, kernels, &num_kernels)
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

cdef void _finish(CLCommandQueue queue) except *:
    cdef cl_int errcode
    errcode = clFinish(queue._command_queue)
    if errcode < 0: raise CLError(error_translation_table[errcode])

cdef void _flush(CLCommandQueue queue) except *:
    cdef cl_int errcode
    errcode = clFlush(queue._command_queue)
    if errcode < 0: raise CLError(error_translation_table[errcode])


cdef void _build(CLProgram program, list options) except *:
    cdef cl_int errcode = clBuildProgram(program._program, 0, NULL, NULL, NULL, NULL)
    if errcode < 0: raise CLError(error_translation_table[errcode])

${makesection("Classes")}
cdef class CLCommandQueue(CLObject):
${make_dealloc("clReleaseCommandQueue(self._command_queue)")}
    def flush(self):
        _flush(self)

    def finish(self):
        _finish(self)

cdef class CLProgram(CLObject):
${make_dealloc("clReleaseProgram(self._program)")}
    def createKernelsInProgram(self):
        return _createKernelsInProgram(self)

    def createKernel(self, bytes string):
        return _createKernel(self, string)

    def getBuildLog(self, CLDevice device):
        return _getBuildLog(self, device)

cdef class CLDevice(CLObject):
${properties_getter("Device", "_device", device_properties)}
${properties_repr(['name', 'type', 'vendor', 'driverVersion'])}

cdef class CLPlatform(CLObject):
${properties_getter("Platform", "_platform", platform_properties)}
    def getDevices(self, cl_device_type dtype = 0xFFFFFFFF):
        return _getDevices(self._platform, dtype)
    def build(self, list options = []):
        _build(self, options)
${properties_repr(['name', 'vendor', 'version'])}

cdef class CLBuffer(CLObject):
${properties_getter("Buffer", "_mem", buffer_properties)}
${properties_repr(['size', 'offset'])}
    property offset:
        def __get__(self):
            return self._offset
${make_dealloc("clReleaseMemObject(self._mem)")}


cdef class CLImage(CLBuffer):
${properties_getter("Image", "_mem", image_properties)}
${properties_repr(['shape'])}


cdef class CLKernel(CLObject):
${properties_getter("Kernel", "_kernel", kernel_properties)}
${properties_repr(['name', 'numargs'])}
${make_dealloc("clReleaseKernel(self._kernel)")}


cdef class CLEvent(CLObject):
${make_dealloc("clReleaseEvent(self._event)")}
${properties_getter("Event", "_event", event_properties)}
${properties_getter("EventProfiling", "_event", profiling_properties)}


cdef class CLSampler(CLObject):
${make_dealloc("clReleaseSampler(self._sampler)")}
${properties_getter("Sampler", "_sampler", sampler_properties)}
${properties_repr(['normalized', 'filterMode', 'addressingMode'])}

cdef class CLContext(CLObject):
${make_dealloc("clReleaseContext (self._context)")}
${properties_repr([])}
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


${makesection("Module level API")}
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
    cdef cl_int errcode
    cdef CLContext instance

    for i from 0 <= i < min(num_devices, 100):
        clDevices[i] = (<CLDevice>devices[i])._device
    context = clCreateContext(NULL, num_devices, clDevices,
                              NULL, NULL, &errcode)
    if errcode < 0: raise translateError(errcode)
    instance = CLContext.__new__(CLContext)
    ${init_instance(['context', 'devices'], '    ')}
    return instance

cpdef waitForEvents(list events):
    cdef int num_events = len(events)
    cdef cl_event lst[100]
    cdef CLEvent evt
    cdef cl_int errcode

    for i from 0 <= i < min(num_events, 100):
        lst[i] = (<CLEvent>events[i])._event
    errcode = clWaitForEvents(num_events, lst)
    if errcode < 0: raise translateError(errcode)

<%def name="makesection(name)">
#
#
#   ${name}
#
#
</%def>

<%def name="copyright()">
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
</%def>

<%def name="info_getter(name, fct_name, obj_type, info_type, return_types)">
    %for t in return_types:
cdef ${t} _${name}_${t}(${obj_type} obj, ${info_type} param_name):
    cdef size_t size
    %if t != "bytes":
    cdef ${t} result
    cdef cl_int errcode = ${fct_name}(obj, param_name, sizeof(${t}), &result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return result
    %else:
    cdef char result[256]
    cdef cl_int errcode = ${fct_name}(obj, param_name, 256 * sizeof(char), result, &size)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    cdef bytes s = result[:size -1]
    return s
    %endif

    %endfor
</%def>
<%def name="properties_getter(obj_type, internal, properties_desc)">
%for ptype in properties_desc:
    %for pname, pdefine in properties_desc[ptype]:
    property ${pname}:
        def __get__(self):
        %if isinstance(pdefine, tuple):
<%ret_string = "" %>\
            %for i, define in enumerate(pdefine):
                cdef ${ptype} r_${i} = _get${obj_type}Info_${ptype}(self.${internal},
                                        ${define})
<%ret_string += "r_%d, " % i %>\
            %endfor
                return (${ret_string})

        %else:
            return _get${obj_type}Info_${ptype}(self.${internal},
                                        ${pdefine})
        %endif
    %endfor
%endfor
</%def>

<%def name="properties_repr(attributes)">
<%repr_str = "<%s"%>\
<%prop_str = "self.__class__.__name__,"%>\
%for pname in attributes:
<%repr_str += " " + pname + '="%s"' %>\
<%prop_str += " self." + pname + "," %>\
%endfor
<%repr_str += ">" %>\
    def __repr__(self):
        return '${repr_str}' % (${prop_str} )
</%def>
<%def name="init_instance(args, indentation)">
    %for a in args: 
${indentation}instance._${a} = ${a}
    %endfor
</%def>
<%def name="make_dealloc(command)">
    def __dealloc__(self):
        cdef cl_int errcode
        errcode = ${command} \
<%text>
        if errcode < 0: print("Error in OpenCL deallocation <%s>" % self.__class__.__name__)
</%text>\
</%def>


