from opencl cimport *
from libopencl cimport *

cimport numpy as np

cdef class CLCommand:
    cdef object call(self, CLCommandQueue queue)

cdef class CLCopyBuffer(CLCommand):
    cdef CLBuffer       _src
    cdef CLBuffer       _dst
    cdef size_t         _cb

    cdef object call(self, CLCommandQueue queue)

cdef class CLReadBufferNDArray(CLCommand):
    cdef CLBuffer       _src
    cdef np.ndarray     _dst
    cdef size_t         _cb
    cdef cl_bool        _blocking

    cdef object call(self, CLCommandQueue queue)

cdef class CLWriteBufferNDArray(CLCommand):
    cdef CLBuffer       _dst
    cdef np.ndarray     _src
    cdef size_t         _cb
    cdef cl_bool        _blocking

    cdef object call(self, CLCommandQueue queue)

cdef class CLMapBuffer(CLCommand):
    cdef cl_map_flags   _flags
    cdef cl_bool        _blocking
    cdef CLBuffer       _src
    cdef CLMappedBuffer _dst

    cdef object call(self, CLCommandQueue queue)

cdef class CLUnmapBuffer(CLCommand):
    cdef CLMappedBuffer _dst

    cdef object call(self, CLCommandQueue queue)

cdef class CLNDRangeKernel(CLCommand):
    cdef CLKernel       _kernel
    cdef size_t         _gws[3]
    cdef size_t         _lws[3]

    cdef object call(self, CLCommandQueue queue)

cdef class CLReadImageNDArray(CLCommand):
    cdef size_t         _origin[3]
    cdef size_t         _shape[3]
    cdef size_t         _row_pitch
    cdef size_t         _slice_pitch
    cdef np.ndarray     _dst
    cdef CLImage        _src
    cdef cl_bool        _blocking

    cdef object call(self, CLCommandQueue queue)

cdef class CLWriteImageNDArray(CLCommand):
    cdef size_t         _origin[3]
    cdef size_t         _shape[3]
    cdef size_t         _row_pitch
    cdef size_t         _slice_pitch
    cdef np.ndarray     _src
    cdef CLImage        _dst
    cdef cl_bool        _blocking

    cdef object call(self, CLCommandQueue queue)

cdef class CLMarker(CLCommand):
    cdef object call(self, CLCommandQueue queue)

cdef class CLBarrier(CLCommand):
    cdef object call(self, CLCommandQueue queue)
