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


        cdef cl_int clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void *, size_t *) 
        cdef cl_int clGetPlatformIDs(cl_uint, cl_platform_id *, cl_uint *)

        cdef cl_int clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *)
        cdef cl_int clGetPlatformInfo(cl_platform_id, cl_platform_info, size_t, void *, size_t *)
        cdef cl_int clGetDeviceInfo(cl_device_id, cl_device_info, size_t, void *, size_t *)

        #cdef cl_context clCreateContext(cl_context_properties *, cl_uint, cl_device_id *, void *pfn_notify (char *, void *, size_t, void *),void *, cl_int *)
        cdef cl_context clCreateContext(cl_context_properties *, cl_uint, cl_device_id *, void *,void *, cl_int *)
        cdef cl_int clReleaseContext(cl_context)
        cdef cl_mem clCreateBuffer(cl_context context, cl_mem_flags, size_t, void *, cl_int *)
        cdef cl_int clReleaseMemObject(cl_mem)


