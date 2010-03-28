<%
param_types = ['byte', 'ubyte',
               'short', 'ushort',
               'int32', 'uint32',
               'int64', 'uint64',
               'intp',
               'float32',
               'float64',
               'float128'];

error_types = ['CL_DEVICE_TYPE',
              'CL_DEVICE_VENDOR_ID',
              'CL_DEVICE_MAX_COMPUTE_UNITS',
              'CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS',
              'CL_DEVICE_MAX_WORK_GROUP_SIZE',
              'CL_DEVICE_MAX_WORK_ITEM_SIZES',
              'CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR',
              'CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT',
              'CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT',
              'CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG',
              'CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT',
              'CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE',
              'CL_DEVICE_MAX_CLOCK_FREQUENCY',
              'CL_DEVICE_ADDRESS_BITS',
              'CL_DEVICE_MAX_READ_IMAGE_ARGS',
              'CL_DEVICE_MAX_WRITE_IMAGE_ARGS',
              'CL_DEVICE_MAX_MEM_ALLOC_SIZE',
              'CL_DEVICE_IMAGE2D_MAX_WIDTH',
              'CL_DEVICE_IMAGE2D_MAX_HEIGHT',
              'CL_DEVICE_IMAGE3D_MAX_WIDTH',
              'CL_DEVICE_IMAGE3D_MAX_HEIGHT',
              'CL_DEVICE_IMAGE3D_MAX_DEPTH',
              'CL_DEVICE_IMAGE_SUPPORT',
              'CL_DEVICE_MAX_PARAMETER_SIZE',
              'CL_DEVICE_MAX_SAMPLERS',
              'CL_DEVICE_MEM_BASE_ADDR_ALIGN',
              'CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE',
              'CL_DEVICE_SINGLE_FP_CONFIG',
              'CL_DEVICE_GLOBAL_MEM_CACHE_TYPE',
              'CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE',
              'CL_DEVICE_GLOBAL_MEM_CACHE_SIZE',
              'CL_DEVICE_GLOBAL_MEM_SIZE',
              'CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE',
              'CL_DEVICE_MAX_CONSTANT_ARGS',
              'CL_DEVICE_LOCAL_MEM_TYPE',
              'CL_DEVICE_LOCAL_MEM_SIZE',
              'CL_DEVICE_ERROR_CORRECTION_SUPPORT',
              'CL_DEVICE_PROFILING_TIMER_RESOLUTION',
              'CL_DEVICE_ENDIAN_LITTLE',
              'CL_DEVICE_AVAILABLE',
              'CL_DEVICE_COMPILER_AVAILABLE',
              'CL_DEVICE_EXECUTION_CAPABILITIES',
              'CL_DEVICE_QUEUE_PROPERTIES',
              'CL_DEVICE_NAME',
              'CL_DEVICE_VENDOR',
              'CL_DRIVER_VERSION',
              'CL_DEVICE_PROFILE',
              'CL_DEVICE_VERSION',
              'CL_DEVICE_EXTENSIONS',
              'CL_DEVICE_PLATFORM']

device_properties = { 'bytes'  :  [('driverVersion',            'CL_DRIVER_VERSION'),
                                   ('vendor',                   'CL_DEVICE_VERSION'),
                                   ('version',                  'CL_DEVICE_VENDOR',),
                                   ('profile',                  'CL_DRIVER_PROFILE'),
                                   ('name',                     'CL_DEVICE_NAME'),
                                   ('extensions',               'CL_DEVICE_EXTENSIONS')],
                     'cl_uint'  : [('addressBits',              'CL_DEVICE_ADDRESS_BITS'),
                                   ('vendorId',                 'CL_DEVICE_VENDOR_ID'),
                                   ('maxClockFrequency',        'CL_DEVICE_MAX_CLOCK_FREQUENCY'),
                                   ('maxComputeUnit',           'CL_DEVICE_MAX_COMPUTE_UNIT'),
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

${makesection("Helper functions")}
cdef ptype param_converter_array[${len(param_types)}]

%for i, type in enumerate(param_types):
cdef param from_${type}(object val) except *:
    cdef param p
    p.${type}_value = <np.npy_${type}>val
    return p

param_converter_array[${i}].itemsize = sizeof(np.npy_${type})
param_converter_array[${i}].fct = from_${type}

%endfor
${info_getter("getDeviceInfo","clGetDeviceInfo", "cl_device_id", "cl_device_info", device_properties)}
${info_getter("getPlatformInfo","clGetPlatformInfo", "cl_platform_id", "cl_platform_info", platform_properties)}
${info_getter("getBufferInfo","clGetMemObjectInfo", "cl_mem", "cl_mem_info", buffer_properties)}
${info_getter("getImageInfo","clGetImageInfo", "cl_mem", "cl_image_info", image_properties)}
${info_getter("getKernelInfo","clGetKernelInfo", "cl_kernel", "cl_kernel_info", kernel_properties)}
${info_getter("getEventInfo","clGetEventInfo", "cl_event", "cl_event_info", event_properties)}
${info_getter("getEventProfilingInfo","clGetEventProfilingInfo", "cl_event", "cl_profiling_info", profiling_properties)}
${info_getter("getSamplerInfo","clGetSamplerInfo", "cl_sampler", "cl_sampler_info", sampler_properties)}

${makesection("Classes")}
cdef class CLDevice(CLObject):
${properties_getter("Device", "_device", device_properties)}

cdef class CLPlatform(CLObject):
${properties_getter("Platform", "_platform", platform_properties)}

cdef class CLBuffer(CLObject):
${properties_getter("Buffer", "_mem", buffer_properties)}

cdef class CLImage(CLBuffer):
${properties_getter("Image", "_mem", image_properties)}

cdef class CLKernel(CLObject):
${properties_getter("Kernel", "_kernel", kernel_properties)}

cdef class CLEvent(CLObject):
${properties_getter("Event", "_event", event_properties)}
${properties_getter("EventProfiling", "_event", profiling_properties)}

cdef class CLSampler(CLObject):
${properties_getter("Sampler", "_sampler", sampler_properties)}

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
<%ret_string = "" %>
            %for i, define in enumerate(pdefine):
                cdef ${ptype} r_${i} = _get${obj_type}Info_${ptype}(self.${internal},
                                        ${define})
<%ret_string += "r_%d, " % i %>
            %endfor
                return (${ret_string})
        %else:
            return _get${obj_type}Info_${ptype}(self.${internal},
                                        ${pdefine})
        %endif
    %endfor
%endfor
</%def>


