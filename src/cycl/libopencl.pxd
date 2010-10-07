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

cdef extern from *:
        ctypedef char * const_char_ptr  "const char *"

cdef extern from "CL/cl.h":
        ctypedef signed char        cl_char
        ctypedef unsigned char      cl_uchar
        ctypedef signed short       cl_short
        ctypedef unsigned short     cl_ushort
        ctypedef signed int         cl_int
        ctypedef unsigned int       cl_uint
        ctypedef signed long long   cl_long
        ctypedef unsigned long long cl_ulong
        ctypedef unsigned short     cl_half
        ctypedef float              cl_float
        ctypedef double             cl_double
        ctypedef cl_uint            cl_bool
        ctypedef cl_int *           cl_context_properties

        ctypedef void *cl_platform_id
        ctypedef void *cl_device_id
        ctypedef void *cl_context
        ctypedef void *cl_command_queue
        ctypedef void *cl_mem
        ctypedef void *cl_program
        ctypedef void *cl_kernel
        ctypedef void *cl_event
        ctypedef void *cl_sampler

        ctypedef cl_ulong       cl_bitfield
        ctypedef cl_bitfield    cl_device_type
        ctypedef cl_uint        cl_platform_info
        ctypedef cl_uint        cl_device_info
        ctypedef cl_bitfield    cl_device_address_info
        ctypedef cl_bitfield    cl_device_fp_config
        ctypedef cl_uint        cl_device_mem_cache_type
        ctypedef cl_uint        cl_device_local_mem_type
        ctypedef cl_bitfield    cl_device_exec_capabilities
        ctypedef cl_bitfield    cl_command_queue_properties
        ctypedef cl_uint        cl_filter_mode
        ctypedef cl_uint        cl_addressing_mode
        ctypedef cl_uint        cl_profiling_info
        ctypedef cl_uint        cl_kernel_work_group_info

        ctypedef enum cl_device_info_enum:
                CL_DEVICE_TYPE                              = 0x1000
                CL_DEVICE_VENDOR_ID                         = 0x1001
                CL_DEVICE_MAX_COMPUTE_UNITS                 = 0x1002
                CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS          = 0x1003
                CL_DEVICE_MAX_WORK_GROUP_SIZE               = 0x1004
                CL_DEVICE_MAX_WORK_ITEM_SIZES               = 0x1005
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR       = 0x1006
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT      = 0x1007
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT        = 0x1008
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG       = 0x1009
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT      = 0x100A
                CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE     = 0x100B
                CL_DEVICE_MAX_CLOCK_FREQUENCY               = 0x100C
                CL_DEVICE_ADDRESS_BITS                      = 0x100D
                CL_DEVICE_MAX_READ_IMAGE_ARGS               = 0x100E
                CL_DEVICE_MAX_WRITE_IMAGE_ARGS              = 0x100F
                CL_DEVICE_MAX_MEM_ALLOC_SIZE                = 0x1010
                CL_DEVICE_IMAGE2D_MAX_WIDTH                 = 0x1011
                CL_DEVICE_IMAGE2D_MAX_HEIGHT                = 0x1012
                CL_DEVICE_IMAGE3D_MAX_WIDTH                 = 0x1013
                CL_DEVICE_IMAGE3D_MAX_HEIGHT                = 0x1014
                CL_DEVICE_IMAGE3D_MAX_DEPTH                 = 0x1015
                CL_DEVICE_IMAGE_SUPPORT                     = 0x1016
                CL_DEVICE_MAX_PARAMETER_SIZE                = 0x1017
                CL_DEVICE_MAX_SAMPLERS                      = 0x1018
                CL_DEVICE_MEM_BASE_ADDR_ALIGN               = 0x1019
                CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE          = 0x101A
                CL_DEVICE_SINGLE_FP_CONFIG                  = 0x101B
                CL_DEVICE_GLOBAL_MEM_CACHE_TYPE             = 0x101C
                CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE         = 0x101D
                CL_DEVICE_GLOBAL_MEM_CACHE_SIZE             = 0x101E
                CL_DEVICE_GLOBAL_MEM_SIZE                   = 0x101F
                CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE          = 0x1020
                CL_DEVICE_MAX_CONSTANT_ARGS                 = 0x1021
                CL_DEVICE_LOCAL_MEM_TYPE                    = 0x1022
                CL_DEVICE_LOCAL_MEM_SIZE                    = 0x1023
                CL_DEVICE_ERROR_CORRECTION_SUPPORT          = 0x1024
                CL_DEVICE_PROFILING_TIMER_RESOLUTION        = 0x1025
                CL_DEVICE_ENDIAN_LITTLE                     = 0x1026
                CL_DEVICE_AVAILABLE                         = 0x1027
                CL_DEVICE_COMPILER_AVAILABLE                = 0x1028
                CL_DEVICE_EXECUTION_CAPABILITIES            = 0x1029
                CL_DEVICE_QUEUE_PROPERTIES                  = 0x102A
                CL_DEVICE_NAME                              = 0x102B
                CL_DEVICE_VENDOR                            = 0x102C
                CL_DRIVER_VERSION                           = 0x102D
                CL_DEVICE_PROFILE                           = 0x102E
                CL_DEVICE_VERSION                           = 0x102F
                CL_DEVICE_EXTENSIONS                        = 0x1030
                CL_DEVICE_PLATFORM                          = 0x1031

        ctypedef enum cl_platform_info_enum:
                CL_PLATFORM_PROFILE                         = 0x0900
                CL_PLATFORM_VERSION                         = 0x0901
                CL_PLATFORM_NAME                            = 0x0902
                CL_PLATFORM_VENDOR                          = 0x0903
                CL_PLATFORM_EXTENSIONS                      = 0x0904

        ctypedef enum cl_device_type_enum:
                CL_DEVICE_TYPE_DEFAULT                      = (1 << 0)
                CL_DEVICE_TYPE_CPU                          = (1 << 1)
                CL_DEVICE_TYPE_GPU                          = (1 << 2)
                CL_DEVICE_TYPE_ACCELERATOR                  = (1 << 3)
                CL_DEVICE_TYPE_ALL                          = 0xFFFFFFFF

        ctypedef enum cl_error_codes_enum:
                CL_SUCCESS                                  = 0
                CL_DEVICE_NOT_FOUND                         = -1
                CL_DEVICE_NOT_AVAILABLE                     = -2
                CL_COMPILER_NOT_AVAILABLE                   = -3
                CL_MEM_OBJECT_ALLOCATION_FAILURE            = -4
                CL_OUT_OF_RESOURCES                         = -5
                CL_OUT_OF_HOST_MEMORY                       = -6
                CL_PROFILING_INFO_NOT_AVAILABLE             = -7
                CL_MEM_COPY_OVERLAP                         = -8
                CL_IMAGE_FORMAT_MISMATCH                    = -9
                CL_IMAGE_FORMAT_NOT_SUPPORTED               = -10
                CL_BUILD_PROGRAM_FAILURE                    = -11
                CL_MAP_FAILURE                              = -12

                CL_INVALID_VALUE                            = -30
                CL_INVALID_DEVICE_TYPE                      = -31
                CL_INVALID_PLATFORM                         = -32
                CL_INVALID_DEVICE                           = -33
                CL_INVALID_CONTEXT                          = -34
                CL_INVALID_QUEUE_PROPERTIES                 = -35
                CL_INVALID_COMMAND_QUEUE                    = -36
                CL_INVALID_HOST_PTR                         = -37
                CL_INVALID_MEM_OBJECT                       = -38
                CL_INVALID_IMAGE_FORMAT_DESCRIPTOR          = -39
                CL_INVALID_IMAGE_SIZE                       = -40
                CL_INVALID_SAMPLER                          = -41
                CL_INVALID_BINARY                           = -42
                CL_INVALID_BUILD_OPTIONS                    = -43
                CL_INVALID_PROGRAM                          = -44
                CL_INVALID_PROGRAM_EXECUTABLE               = -45
                CL_INVALID_KERNEL_NAME                      = -46
                CL_INVALID_KERNEL_DEFINITION                = -47
                CL_INVALID_KERNEL                           = -48
                CL_INVALID_ARG_INDEX                        = -49
                CL_INVALID_ARG_VALUE                        = -50
                CL_INVALID_ARG_SIZE                         = -51
                CL_INVALID_KERNEL_ARGS                      = -52
                CL_INVALID_WORK_DIMENSION                   = -53
                CL_INVALID_WORK_GROUP_SIZE                  = -54
                CL_INVALID_WORK_ITEM_SIZE                   = -55
                CL_INVALID_GLOBAL_OFFSET                    = -56
                CL_INVALID_EVENT_WAIT_LIST                  = -57
                CL_INVALID_EVENT                            = -58
                CL_INVALID_OPERATION                        = -59
                CL_INVALID_GL_OBJECT                        = -60
                CL_INVALID_BUFFER_SIZE                      = -61
                CL_INVALID_MIP_LEVEL                        = -62
                CL_INVALID_GLOBAL_WORK_SIZE                 = -63

        ctypedef enum cl_mem_flags:
                CL_MEM_READ_WRITE                           = (1 << 0)
                CL_MEM_WRITE_ONLY                           = (1 << 1)
                CL_MEM_READ_ONLY                            = (1 << 2)
                CL_MEM_USE_HOST_PTR                         = (1 << 3)
                CL_MEM_ALLOC_HOST_PTR                       = (1 << 4)
                CL_MEM_COPY_HOST_PTR                        = (1 << 5)

        ctypedef enum cl_channel_order:
                CL_R                                        = 0x10B0
                CL_A                                        = 0x10B1
                CL_RG                                       = 0x10B2
                CL_RA                                       = 0x10B3
                CL_RGB                                      = 0x10B4
                CL_RGBA                                     = 0x10B5
                CL_BGRA                                     = 0x10B6
                CL_ARGB                                     = 0x10B7
                CL_INTENSITY                                = 0x10B8
                CL_LUMINANCE                                = 0x10B9

        ctypedef enum cl_channel_type:
                CL_SNORM_INT8                               = 0x10D0
                CL_SNORM_INT16                              = 0x10D1
                CL_UNORM_INT8                               = 0x10D2
                CL_UNORM_INT16                              = 0x10D3
                CL_UNORM_SHORT_565                          = 0x10D4
                CL_UNORM_SHORT_555                          = 0x10D5
                CL_UNORM_INT_101010                         = 0x10D6
                CL_SIGNED_INT8                              = 0x10D7
                CL_SIGNED_INT16                             = 0x10D8
                CL_SIGNED_INT32                             = 0x10D9
                CL_UNSIGNED_INT8                            = 0x10DA
                CL_UNSIGNED_INT16                           = 0x10DB
                CL_UNSIGNED_INT32                           = 0x10DC
                CL_HALF_FLOAT                               = 0x10DD
                CL_FLOAT                                    = 0x10DE

        ctypedef enum cl_image_info:
                CL_IMAGE_FORMAT                             = 0x1110
                CL_IMAGE_ELEMENT_SIZE                       = 0x1111
                CL_IMAGE_ROW_PITCH                          = 0x1112
                CL_IMAGE_SLICE_PITCH                        = 0x1113
                CL_IMAGE_WIDTH                              = 0x1114
                CL_IMAGE_HEIGHT                             = 0x1115
                CL_IMAGE_DEPTH                              = 0x1116

        ctypedef enum cl_mem_info:
                CL_MEM_TYPE                                 = 0x1100
                CL_MEM_FLAGS                                = 0x1101
                CL_MEM_SIZE                                 = 0x1102
                CL_MEM_HOST_PTR                             = 0x1103
                CL_MEM_MAP_COUNT                            = 0x1104
                CL_MEM_REFERENCE_COUNT                      = 0x1105
                CL_MEM_CONTEXT                              = 0x1106

        ctypedef enum cl_program_build_info:
                CL_PROGRAM_BUILD_STATUS                     = 0x1181
                CL_PROGRAM_BUILD_OPTIONS                    = 0x1182
                CL_PROGRAM_BUILD_LOG                        = 0x1183

        ctypedef enum cl_program_info:
                CL_PROGRAM_REFERENCE_COUNT                  = 0x1160
                CL_PROGRAM_CONTEXT                          = 0x1161
                CL_PROGRAM_NUM_DEVICES                      = 0x1162
                CL_PROGRAM_DEVICES                          = 0x1163
                CL_PROGRAM_SOURCE                           = 0x1164
                CL_PROGRAM_BINARY_SIZES                     = 0x1165
                CL_PROGRAM_BINARIES                         = 0x1166

        ctypedef enum cl_kernel_info:
                CL_KERNEL_FUNCTION_NAME                     = 0x1190
                CL_KERNEL_NUM_ARGS                          = 0x1191
                CL_KERNEL_REFERENCE_COUNT                   = 0x1192
                CL_KERNEL_CONTEXT                           = 0x1193
                CL_KERNEL_PROGRAM                           = 0x1194

        ctypedef enum cl_command_type:
                CL_COMMAND_NDRANGE_KERNEL                   = 0x11F0
                CL_COMMAND_TASK                             = 0x11F1
                CL_COMMAND_NATIVE_KERNEL                    = 0x11F2
                CL_COMMAND_READ_BUFFER                      = 0x11F3
                CL_COMMAND_WRITE_BUFFER                     = 0x11F4
                CL_COMMAND_COPY_BUFFER                      = 0x11F5
                CL_COMMAND_READ_IMAGE                       = 0x11F6
                CL_COMMAND_WRITE_IMAGE                      = 0x11F7
                CL_COMMAND_COPY_IMAGE                       = 0x11F8
                CL_COMMAND_COPY_IMAGE_TO_BUFFER             = 0x11F9
                CL_COMMAND_COPY_BUFFER_TO_IMAGE             = 0x11FA
                CL_COMMAND_MAP_BUFFER                       = 0x11FB
                CL_COMMAND_MAP_IMAGE                        = 0x11FC
                CL_COMMAND_UNMAP_MEM_OBJECT                 = 0x11FD
                CL_COMMAND_MARKER                           = 0x11FE
                CL_COMMAND_ACQUIRE_GL_OBJECTS               = 0x11FF
                CL_COMMAND_RELEASE_GL_OBJECTS               = 0x1200

        ctypedef enum cl_addressing_mode:
                CL_ADDRESS_NONE                             = 0x1130
                CL_ADDRESS_CLAMP_TO_EDGE                    = 0x1131
                CL_ADDRESS_CLAMP                            = 0x1132
                CL_ADDRESS_REPEAT                           = 0x1133

        ctypedef enum cl_command_queue_properties:
                CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE      = (1 << 0)
                CL_QUEUE_PROFILING_ENABLE                   = (1 << 1)

        ctypedef enum cl_filter_mode:
                CL_FILTER_NEAREST                          = 0x1140
                CL_FILTER_LINEAR                           = 0x1141

        ctypedef enum cl_sampler_info:
                CL_SAMPLER_REFERENCE_COUNT                  = 0x1150
                CL_SAMPLER_CONTEXT                          = 0x1151
                CL_SAMPLER_NORMALIZED_COORDS                = 0x1152
                CL_SAMPLER_ADDRESSING_MODE                  = 0x1153
                CL_SAMPLER_FILTER_MODE                      = 0x1154

        ctypedef enum cl_event_info:
                CL_EVENT_COMMAND_QUEUE                      = 0x11D0
                CL_EVENT_COMMAND_TYPE                       = 0x11D1
                CL_EVENT_REFERENCE_COUNT                    = 0x11D2
                CL_EVENT_COMMAND_EXECUTION_STATUS           = 0x11D3

        ctypedef enum cl_kernel_work_group_info:
                CL_KERNEL_WORK_GROUP_SIZE                   = 0x11B0
                CL_KERNEL_COMPILE_WORK_GROUP_SIZE           = 0x11B1
                CL_KERNEL_LOCAL_MEM_SIZE                    = 0x11B2

        ctypedef enum cl_profiling_info:
                CL_PROFILING_COMMAND_QUEUED                 = 0x1280
                CL_PROFILING_COMMAND_SUBMIT                 = 0x1281
                CL_PROFILING_COMMAND_START                  = 0x1282
                CL_PROFILING_COMMAND_END                    = 0x1283

        ctypedef enum cl_execution_status:
                CL_COMPLETE                                 = 0x0
                CL_RUNNING                                  = 0x1
                CL_SUBMITTED                                = 0x2
                CL_QUEUED                                   = 0x3

        ctypedef enum cl_map_flags:
                CL_MAP_READ                                 = (1 << 0)
                CL_MAP_WRITE                                = (1 << 1)

        ctypedef enum cl_context_properties:
                CL_CONTEXT_PLATFORM                         = 0x1084

        ctypedef enum cl_kernel_work_group_info:
                CL_KERNEL_WORK_GROUP_SIZE                   = 0x11B0
                CL_KERNEL_COMPILE_WORK_GROUP_SIZE           = 0x11B1
                CL_KERNEL_LOCAL_MEM_SIZE                    = 0x11B2





        ctypedef struct cl_image_format:
                cl_channel_order image_channel_order
                cl_channel_type image_channel_data_type


        # Device API
        cdef cl_int clGetDeviceInfo(cl_device_id,
                                    cl_device_info,
                                    size_t,
                                    void *,
                                    size_t *) nogil

        # Platform API
        cdef cl_int clGetPlatformIDs(cl_uint,
                                     cl_platform_id *,
                                     cl_uint *) nogil

        cdef cl_int clGetDeviceIDs(cl_platform_id,
                                   cl_device_type,
                                   cl_uint,
                                   cl_device_id *,
                                   cl_uint *) nogil

        cdef cl_int clGetPlatformInfo(cl_platform_id,
                                      cl_platform_info,
                                      size_t,
                                      void *,
                                      size_t *) nogil

        cdef cl_context clCreateContext(cl_context_properties *,
                                        cl_uint,
                                        cl_device_id *,
                                        void (*pfn_notify) (char *, void *, size_t, void *),
                                        void *,
                                        cl_int *)

        cdef cl_int clGetContextInfo(cl_context,
                                     cl_context_info,
                                     size_t,
                                     void *,
                                     size_t *)


        # Context API
        #cdef cl_context clCreateContext(cl_context_properties *,
                                        #cl_uint,
                                        #cl_device_id *,
                                        #void *,
                                        #void *,
                                        #cl_int *) nogil

        cdef cl_int clReleaseContext(cl_context) nogil


        # Memory API
        cdef cl_mem clCreateBuffer(cl_context context,
                                   cl_mem_flags,
                                   size_t,
                                   void *,
                                   cl_int *) nogil


        cdef cl_mem clCreateImage2D(cl_context,
                                    cl_mem_flags,
                                    cl_image_format *,
                                    size_t,
                                    size_t,
                                    size_t,
                                    void *,
                                    cl_int *) nogil

        cdef cl_mem clCreateImage3D(cl_context,
                                    cl_mem_flags,
                                    cl_image_format *,
                                    size_t,
                                    size_t,
                                    size_t,
                                    size_t,
                                    size_t,
                                    void *,
                                    cl_int *) nogil

        cdef cl_int clReleaseMemObject(cl_mem) nogil

        cdef cl_int clGetImageInfo(cl_mem,
                                   cl_image_info,
                                   size_t,
                                   void *,
                                   size_t *) nogil

        cdef cl_int clGetMemObjectInfo(cl_mem,
                                       cl_mem_info,
                                       size_t,
                                       void *,
                                       size_t *) nogil
        # Program API
        cdef cl_program clCreateProgramWithSource(cl_context,
                                                  cl_uint,
                                                  char **,
                                                  size_t *,
                                                  cl_int *) nogil

        cdef cl_int clReleaseProgram(cl_program) nogil

        cdef cl_int clBuildProgram(cl_program,
                                   cl_uint,
                                   cl_device_id *,
                                   char *,
                                   void (*pfn_notify)(cl_program, void *user_data),
                                   void *) nogil

        cdef cl_int clCreateKernelsInProgram(cl_program,
                                             cl_uint,
                                             cl_kernel *,
                                             cl_uint *) nogil

        # Kernel API
        cdef cl_int clReleaseKernel(cl_kernel) nogil


        cdef cl_kernel clCreateKernel(cl_program,
                                      char *,
                                      cl_int *) nogil

        cdef cl_int clGetProgramBuildInfo(cl_program,
                                          cl_device_id,
                                          cl_program_build_info,
                                          size_t,
                                          void *,
                                          size_t *) nogil
        cdef cl_int clGetProgramInfo(cl_program program,
                                     cl_program_info,
                                     size_t,
                                     void *,
                                     size_t *) nogil


        cdef cl_int clGetKernelInfo(cl_kernel,
                                    cl_kernel_info,
                                    size_t,
                                    void *,
                                    size_t *) nogil

        cdef cl_int clGetKernelWorkGroupInfo(cl_kernel,
                                             cl_device_id,
                                             cl_kernel_work_group_info,
                                             size_t,
                                             void *,
                                             size_t *) nogil

        cdef cl_int clSetKernelArg(cl_kernel,
                                   cl_uint,
                                   size_t,
                                   void *) nogil

        # Command Queue API
        cdef cl_command_queue clCreateCommandQueue(cl_context,
                                                   cl_device_id,
                                                   cl_command_queue_properties,
                                                   cl_int *) nogil

        cdef cl_int clReleaseCommandQueue(cl_command_queue) nogil

        cdef cl_int clFinish(cl_command_queue) nogil

        cdef cl_int clFlush(cl_command_queue) nogil

        cdef cl_int clEnqueueNDRangeKernel(cl_command_queue,
                                           cl_kernel,
                                           cl_uint,
                                           size_t *,
                                           size_t *,
                                           size_t *,
                                           cl_uint,
                                           cl_event *,
                                           cl_event *) nogil


        cdef cl_int clEnqueueCopyImageToBuffer(cl_command_queue,
                                               cl_mem,
                                               cl_mem,
                                               size_t *,
                                               size_t *,
                                               size_t,
                                               cl_uint,
                                               cl_event *,
                                               cl_event *) nogil

        cdef cl_int clEnqueueCopyBufferToImage(cl_command_queue,
                                               cl_mem,
                                               cl_mem,
                                               size_t,
                                               size_t *,
                                               size_t *,
                                               cl_uint,
                                               cl_event *,
                                               cl_event *) nogil


        cdef cl_int clEnqueueWriteBuffer(cl_command_queue,
                                         cl_mem,
                                         cl_bool,
                                         size_t,
                                         size_t,
                                         void *,
                                         cl_uint,
                                         cl_event *,
                                         cl_event *) nogil

        cdef cl_int clEnqueueReadBuffer(cl_command_queue,
                                        cl_mem,
                                        cl_bool,
                                        size_t,
                                        size_t,
                                        void *,
                                        cl_uint,
                                        cl_event *,
                                        cl_event *) nogil

        cdef cl_int clEnqueueCopyBuffer(cl_command_queue,
                                        cl_mem,
                                        cl_mem,
                                        size_t,
                                        size_t,
                                        size_t,
                                        cl_uint,
                                        cl_event *,
                                        cl_event *) nogil

        cdef void * clEnqueueMapBuffer (cl_command_queue command_queue,
                                        cl_mem buffer,
                                        cl_bool blocking_map,
                                        cl_map_flags map_flags,
                                        size_t offset,
                                        size_t cb,
                                        cl_uint num_events_in_wait_list,
                                        cl_event *event_wait_list,
                                        cl_event *event,
                                        cl_int *errcode_ret)

        cdef cl_int clEnqueueBarrier(cl_command_queue) nogil

        cdef cl_int clEnqueueMarker(cl_command_queue, cl_event *) nogil

        cdef cl_int clEnqueueWaitForEvents(cl_command_queue,
                                           cl_uint,
                                           cl_event *) nogil

        # Sampler API
        cdef cl_int clReleaseSampler(cl_sampler) nogil

        cdef cl_sampler clCreateSampler(cl_context,
                                        cl_bool,
                                        cl_addressing_mode,
                                        cl_filter_mode,
                                        cl_int *) nogil

        cdef cl_int clGetSamplerInfo(cl_sampler,
                                     cl_sampler_info,
                                     size_t,
                                     void *,
                                     size_t *) nogil

        # Event API
        cdef cl_int clReleaseEvent(cl_event) nogil

        cdef cl_int clGetEventInfo(cl_event,
                                   cl_event_info,
                                   size_t,
                                   void *,
                                   size_t *) nogil

        cdef cl_int clWaitForEvents(cl_uint,
                                    cl_event *) nogil

        cdef cl_int clGetEventProfilingInfo(cl_event,
                                            cl_profiling_info,
                                            size_t,
                                            void *,
                                            size_t *) nogil

        cdef cl_int clEnqueueUnmapMemObject(cl_command_queue,
                                            cl_mem,
                                            void *,
                                            cl_uint,
                                            cl_event *,
                                            cl_event *) nogil

        cdef cl_int clEnqueueReadImage (cl_command_queue queue,
                                        cl_mem image,
                                        cl_bool blocking,
                                        size_t origin[3],
                                        size_t region[3],
                                        size_t row_pitch,
                                        size_t slice_pitch,
                                        void *ptr,
                                        cl_uint num_events_in_wait_list,
                                        cl_event *event_wait_list,
                                        cl_event *event) nogil


        cdef cl_int clEnqueueWriteImage (cl_command_queue command_queue,
                                         cl_mem image,
                                         cl_bool blocking_write,
                                         size_t origin[3],
                                         size_t region[3],
                                         size_t input_row_pitch,
                                         size_t input_slice_pitch,
                                         void *ptr,
                                         cl_uint num_events_in_wait_list,
                                         cl_event *event_wait_list,
                                         cl_event *event) nogil




