from libopencl cimport *

cdef extern from "pyerrors.h":
    ctypedef class __builtin__.Exception [object PyBaseExceptionObject]: pass

cdef class CLError(Exception): pass
cdef CLError translateError(cl_int error)


cdef class CLObject: pass

cdef class CLDevice(CLObject):
    cdef cl_device_id _device
