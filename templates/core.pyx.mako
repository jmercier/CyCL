<%namespace file="functions.mako" import="*"/>\
${copyright()}
<%
from itertools import izip
param_types = ['char', 'uchar',
               'short', 'ushort',
               'int', 'uint',
               'long', 'ulong',
               'half',
               'float',
               'double',
               'bool',
               #'float128' # UNTIL WARNING IN GCC
              ];

defines_types = { 'memory_flag' :  ['CL_MEM_READ_WRITE',
                                    'CL_MEM_WRITE_ONLY',
                                    'CL_MEM_READ_ONLY',
                                    'CL_MEM_USE_HOST_PTR',
                                    'CL_MEM_ALLOC_HOST_PTR',
                                    'CL_MEM_COPY_HOST_PTR'],
                 'filter_mode' :   ['CL_FILTER_NEAREST',
                                    'CL_FILTER_LINEAR'],
                 'event_status':   ['CL_COMPLETE',
                                    'CL_RUNNING',
                                    'CL_SUBMITTED',
                                    'CL_QUEUED'],
                 'device_type' :   ['CL_DEVICE_TYPE_DEFAULT',
                                    'CL_DEVICE_TYPE_CPU',
                                    'CL_DEVICE_TYPE_GPU',
                                    'CL_DEVICE_TYPE_ACCELERATOR',
                                    'CL_DEVICE_TYPE_ALL'],
                 'channel_type':   ['CL_SNORM_INT8',
                                    'CL_SNORM_INT16',
                                    'CL_UNORM_INT8',
                                    'CL_UNORM_INT16',
                                    'CL_UNORM_SHORT_565',
                                    'CL_UNORM_SHORT_555',
                                    'CL_UNORM_INT_101010',
                                    'CL_SIGNED_INT8',
                                    'CL_SIGNED_INT16',
                                    'CL_SIGNED_INT32',
                                    'CL_UNSIGNED_INT8',
                                    'CL_UNSIGNED_INT16',
                                    'CL_UNSIGNED_INT32',
                                    'CL_HALF_FLOAT',
                                    'CL_FLOAT' ],
                 'channel_order':  ['CL_R',
                                    'CL_A',
                                    'CL_RG',
                                    'CL_RA',
                                    'CL_RGB',
                                    'CL_RGBA',
                                    'CL_BGRA',
                                    'CL_ARGB',
                                    'CL_INTENSITY',
                                    'CL_LUMINANCE'],
                 'addressing_mode':['CL_ADDRESS_NONE',
                                    'CL_ADDRESS_CLAMP_TO_EDGE',
                                    'CL_ADDRESS_CLAMP',
                                    'CL_ADDRESS_REPEAT'],
                 'command_type'  : ['CL_COMMAND_NDRANGE_KERNEL',
                                    'CL_COMMAND_TASK',
                                    'CL_COMMAND_NATIVE_KERNEL',
                                    'CL_COMMAND_READ_BUFFER',
                                    'CL_COMMAND_WRITE_BUFFER',
                                    'CL_COMMAND_COPY_BUFFER',
                                    'CL_COMMAND_READ_IMAGE',
                                    'CL_COMMAND_WRITE_IMAGE',
                                    'CL_COMMAND_COPY_IMAGE',
                                    'CL_COMMAND_COPY_IMAGE_TO_BUFFER',
                                    'CL_COMMAND_COPY_BUFFER_TO_IMAGE',
                                    'CL_COMMAND_MAP_BUFFER',
                                    'CL_COMMAND_MAP_IMAGE',
                                    'CL_COMMAND_UNMAP_MEM_OBJECT',
                                    'CL_COMMAND_MARKER',
                                    'CL_COMMAND_ACQUIRE_GL_OBJECTS',
                                    'CL_COMMAND_RELEASE_GL_OBJECTS'],
                'command_queue_flag' : ['CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE',
                                        'CL_QUEUE_PROFILING_ENABLE']

                }




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

device_properties = \
        { 'bytes'    : [('driverVersion',                'CL_DRIVER_VERSION'),
                        ('vendor',                       'CL_DEVICE_VERSION'),
                        ('version',                      'CL_DEVICE_VENDOR',),
                        ('profile',                      'CL_DEVICE_PROFILE'),
                        ('name',                         'CL_DEVICE_NAME'),
                        ('extensions',                   'CL_DEVICE_EXTENSIONS')],
          'cl_uint'  : [('addressBits',                  'CL_DEVICE_ADDRESS_BITS'),
                        ('vendorId',                     'CL_DEVICE_VENDOR_ID'),
                        ('maxClockFrequency',            'CL_DEVICE_MAX_CLOCK_FREQUENCY'),
                        ('maxComputeUnits',              'CL_DEVICE_MAX_COMPUTE_UNITS'),
                        ('maxWorkItemDimensions',        'CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS'),
                        ('maxConstantArgs',              'CL_DEVICE_MAX_CONSTANT_ARGS'),
                        ('minDataTypeAlignSize',         'CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE'),
                        ('maxWriteImageArgs',            'CL_DEVICE_MAX_WRITE_IMAGE_ARGS'),
                        ('memBaseAddrAlign',             'CL_DEVICE_MEM_BASE_ADDR_ALIGN'),
                        ('preferredVectorWidthChar',     'CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR'),
                        ('preferredVectorWidthShort',    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT'),
                        ('preferredVectorWidthInt',      'CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT'),
                        #('maxWorkItemSizes',            'CL_DEVICE_MAX_WORK_ITEM_SIZES'),
                        ('preferredVectorWidthLong',     'CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG'),
                        ('preferredVectorWidthFloat',    'CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT'),
                        ('preferredVectorWidthDouble',   'CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE')],
          'size_t'   : [('maxWorkGroupSize',             'CL_DEVICE_MAX_WORK_GROUP_SIZE'),
                        ('profilingTimerResolution',     'CL_DEVICE_PROFILING_TIMER_RESOLUTION'),
                        ('image2DMaxSize',               ('CL_DEVICE_IMAGE2D_MAX_HEIGHT',
                                                          'CL_DEVICE_IMAGE2D_MAX_WIDTH')),
                        ('image3DMaxSize',               ('CL_DEVICE_IMAGE3D_MAX_HEIGHT',
                                                          'CL_DEVICE_IMAGE3D_MAX_WIDTH',
                                                          'CL_DEVICE_IMAGE3D_MAX_DEPTH'))],

          'cl_ulong' : [('globalMemSize',                'CL_DEVICE_GLOBAL_MEM_SIZE'),
                        ('globalMemCacheSize',           'CL_DEVICE_GLOBAL_MEM_CACHE_SIZE'),
                        ('globalMemCachelineSize',       'CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE'),
                        ('maxConstantBufferSize',        'CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE'),
                        ('maxMemAllocSize',              'CL_DEVICE_MAX_MEM_ALLOC_SIZE'),
                        ('type',                         'CL_DEVICE_TYPE')],
          'cl_bool'  : [('imageSupport',                 'CL_DEVICE_IMAGE_SUPPORT'),
                        ('ECCSupport',                   'CL_DEVICE_ERROR_CORRECTION_SUPPORT'),
                        ('endianLittle',                 'CL_DEVICE_ENDIAN_LITTLE'),
                        ('compilerAvailable',            'CL_DEVICE_COMPILER_AVAILABLE'),
                        ('available',                    'CL_DEVICE_AVAILABLE')]}


platform_properties = \
        { 'bytes'    : [('version',                      'CL_PLATFORM_VERSION'),
                        ('name',                         'CL_PLATFORM_NAME'),
                        ('vendor',                       'CL_PLATFORM_VENDOR'),
                        ('extensions',                   'CL_PLATFORM_EXTENSIONS'),
                        ('profile',                      'CL_PLATFORM_PROFILE')]}

buffer_properties = \
        { 'size_t'   : [('size',                         'CL_MEM_SIZE')]}

image_properties = \
        { 'size_t'   : [('slicePitch',                   'CL_IMAGE_SLICE_PITCH'),
                        ('elementSize',                  'CL_IMAGE_ELEMENT_SIZE'),
                        ('shape',                       ('CL_IMAGE_WIDTH',
                                                         'CL_IMAGE_HEIGHT',
                                                         'CL_IMAGE_DEPTH')),
                        ('rowPitch',                     'CL_IMAGE_ROW_PITCH')]}

kernel_properties = \
        { 'bytes'    : [('name',                         'CL_KERNEL_FUNCTION_NAME')],
          'cl_uint'  : [('numArgs',                      'CL_KERNEL_NUM_ARGS')]}

event_properties = \
        { 'cl_int'   : [('type',                         'CL_EVENT_COMMAND_TYPE'),
                        ('status',                       'CL_EVENT_COMMAND_EXECUTION_STATUS')]}

profiling_properties = \
        { 'cl_ulong' : [('profilingQueued',              'CL_PROFILING_COMMAND_QUEUED'),
                        ('profilingSubmit',              'CL_PROFILING_COMMAND_SUBMIT'),
                        ('profilingStart',               'CL_PROFILING_COMMAND_START'),
                        ('profilingEnd',                 'CL_PROFILING_COMMAND_END')]}

sampler_properties = \
        { 'cl_uint'  : [('normalized',                   'CL_SAMPLER_NORMALIZED_COORDS'),
                                                        ('filterMode', 'CL_SAMPLER_FILTER_MODE'),
                                                        ('addressingMode', 'CL_SAMPLER_ADDRESSING_MODE')]}

%>\
cdef dict error_translation_table = {
%for e in error_types:
        ${e + ' ' * (40 - len(e))} : "${e}",
%endfor
}


cdef CLError translateError(cl_int errcode):
    return CLError(error_translation_table[errcode])

${makesection("Args Translation")}
cdef union param:
    cl_mem              mem_value
    cl_sampler          sampler_value
%for t in param_types:
    cl_${t} ${' ' * (12 - len(t)) + t}_value
%endfor

ctypedef param (*param_converter_fct)(object) except *

cdef struct ptype:
    size_t                  itemsize
    param_converter_fct     fct

DEF MAX_ARG_TRANSLATION = ${len(param_types) + 2}
cdef ptype param_converter_array[MAX_ARG_TRANSLATION]

%for i, t in enumerate(param_types):
cdef param from_${t}(object val) except *:
    cdef param p
    p.${t}_value = <cl_${t}>val
    return p
param_converter_array[${i}].itemsize = sizeof(cl_${t})
param_converter_array[${i}].fct = from_${t}

%endfor
cdef param from_CLBuffer(object val) except *:
    cdef CLBuffer buf_val = val
    cdef param p
    p.mem_value = buf_val._mem
    return p
param_converter_array[${len(param_types)}].itemsize = sizeof(cl_mem)
param_converter_array[${len(param_types)}].fct = from_CLBuffer

cdef param from_CLSampler(object val) except *:
    cdef CLSampler buf_val = val
    cdef param p
    p.sampler_value = buf_val._sampler
    return p
param_converter_array[${len(param_types) + 1}].itemsize = sizeof(cl_sampler)
param_converter_array[${len(param_types) + 1}].fct = from_CLSampler

class parameter_type(CLObject):
%for i, t in enumerate(param_types + ['MEM', 'SAMPLER']):
    ${t.upper()}_TYPE ${' ' * (12 - len(t))} = ${i}
%endfor
    IMAGE_TYPE         = MEM_TYPE

%for dtype in defines_types:
class ${dtype}(CLObject):
%for f in defines_types[dtype]:
    ${f[3:]} ${' ' * (35 - len(f))} = ${f}
%endfor

%endfor


${makesection("Classes")}
cdef class CLCommandQueue(CLObject):
    ${make_dealloc("clReleaseCommandQueue(self._command_queue)")}
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
    ${make_dealloc("clReleaseProgram(self._program)")}
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

    ${properties_repr(['address', 'mapped'])}
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
    ${properties_getter2("clGetDeviceInfo", "_device", device_properties)}
    ${properties_repr(['name', 'type', 'version'])}


cdef class CLPlatform(CLObject):
    ${properties_getter2("clGetPlatformInfo", "_platform", platform_properties)}
    ${properties_repr(['name', 'vendor', 'version'])}
    def getDevices(self, cl_device_type dtype = 0xFFFFFFFF):
        cdef cl_device_id devices[10]
        cdef cl_uint num_devices
        cdef cl_int errcode = clGetDeviceIDs(self._platform, dtype, 10,
                                             devices, &num_devices)
        if errcode < 0: raise CLError(error_translation_table[errcode])
        return [_createCLDevice(devices[i]) for i from 0 <= i < num_devices]



cdef class CLBuffer(CLObject):
    ${make_dealloc("clReleaseMemObject(self._mem)")}
    ${properties_getter2("clGetMemObjectInfo", "_mem", buffer_properties)}
    ${properties_repr(['size', 'offset'])}
    property offset:
        def __get__(self):
            return self._offset


cdef class CLImage(CLBuffer):
    ${properties_getter2("clGetImageInfo", "_mem", image_properties)}
    ${properties_repr(['shape'])}


cdef class CLKernel(CLObject):
    ${make_dealloc("clReleaseKernel(self._kernel)")}
    ${properties_getter2("clGetKernelInfo", "_kernel", kernel_properties)}
    ${properties_repr(['name', 'numArgs', 'ready'])}
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
    ${make_dealloc("clReleaseEvent(self._event)")}
    ${properties_getter2("clGetEventInfo", "_event", event_properties)}
    ${properties_getter2("clGetEventProfilingInfo", "_event", profiling_properties)}
    ${properties_repr(['type', 'status'])}


cdef class CLSampler(CLObject):
    ${make_dealloc("clReleaseSampler(self._sampler)")}
    ${properties_repr(['normalized', 'filterMode', 'addressingMode'])}
    ${properties_getter2("clGetSamplerInfo", "_sampler", sampler_properties)}


cdef class CLContext(CLObject):
    """
    This object represent a CLContext.
    """
    ${make_dealloc("clReleaseContext (self._context)")}
    ${properties_repr(['devices'])}
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


${makesection("Module level API")}
cpdef list getPlatforms():
    """
    Obtain the list of platforms available.
    """
    cdef cl_platform_id platforms[15]
    cdef cl_uint num_platforms
    cdef cl_int errcode = clGetPlatformIDs(15, platforms, &num_platforms)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return [_createCLPlatform(platforms[i]) for i from 0 <= i < num_platforms]


cpdef CLContext createContext(list devices):
    """
    Creates an OpenCL context.
    """
    cdef long num_devices = len(devices)
    cdef cl_device_id clDevices[100]
    cdef cl_int errcode
    for i from 0 <= i < min(num_devices, 100):
        clDevices[i] = (<CLDevice>devices[i])._device
    cdef cl_context context = clCreateContext(NULL, num_devices, clDevices,
                                              NULL, NULL, &errcode)
    if errcode < 0: raise CLError(error_translation_table[errcode])
    return _createCLContext(devices, context)


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





