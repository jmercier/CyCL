<%namespace file="functions.mako" import="*"/>
${copyright()}

cdef class CLCopyBuffer(CLCommand):
    """
    This command copy a device buffer to another device buffer
    """
    def __cinit__(self, CLBuffer dst, CLBuffer src):
        self._src = src
        self._dst = dst
        self._cb = src.size - src._offset
        if self._cb != dst.size - dst._offset:
            raise AttributeError("Size Mismatch")

    cdef object call(self, CLCommandQueue queue):
        cdef cl_event event
        cdef cl_int errcode
        errcode = clEnqueueCopyBuffer(queue._command_queue,
                                      self._src._mem, self._dst._mem,
                                      self._src_offset, self._dst_offset,
                                      self.cb,
                                      0, NULL,
                                      &event)
        if errcode < 0: raise translateError(errcode)
        return _createCLEvent(event, queue)

cdef class CLReadBufferNDArray(CLCommand):
    """
    This command copy a device buffer to a buffer pointed by
    a numpy array
    """
    def __cinit__(self, np.ndarray dst, CLBuffer src, cl_bool blocking = True):
        self._src = src
        self._dst = dst
        self._cb = src.size - src._offset
        self._blocking = blocking
        if self._cb != dst.size * dst.dtype.itemsize:
            raise AttributeError("Size mismatch")

    cdef object call(self, CLCommandQueue queue):
        cdef cl_event event
        cdef cl_int errcode
        errcode = clEnqueueReadBuffer(queue._command_queue,
                                      self._src._mem,
                                      self._blocking,
                                      self._src._offset,
                                      self._cb,
                                      self._dst.data,
                                      0, NULL,
                                      &event)
        if errcode < 0: raise translateError(errcode)
        return _createCLEvent(event, queue)

cdef class CLNDRangeKernel(CLCommand):
    """
    This command enqueue the execution of a kernel on a device
    with a global and a local working size
    """
    def __cinit__(self, CLKernel kernel,
                  tuple global_work_size = (1,1,1),
                  tuple local_work_size = (1,1,1)):
        self._kernel = kernel
        self._gws[0] = global_work_size[0]
        self._gws[1] = global_work_size[1]
        self._gws[2] = global_work_size[2]
        self._lws[0] = local_work_size[0]
        self._lws[1] = local_work_size[1]
        self._lws[2] = local_work_size[2]


    cdef object call(self, CLCommandQueue queue):
        cdef cl_event event
        cdef cl_int errcode
        errcode = clEnqueueNDRangeKernel(queue._command_queue,
                                         self._kernel._kernel,
                                         3, NULL,
                                         self._gws, self._lws,
                                         0, NULL,
                                         &event)
        if errcode < 0: raise translateError(errcode)
        return _createCLEvent(event, queue)

cdef class CLWriteBufferNDArray(CLCommand):
    """
    This command enqueue the copy of a local memory pointed by a numpy
    array to a device memory buffer
    """
    def __cinit__(self, CLBuffer dst, np.ndarray src, cl_bool blocking = True):
        self._src = src
        self._dst = dst
        self._cb = src.size * src.dtype.itemsize
        self._blocking = blocking
        if self._cb != dst.size - dst._offset:
            raise AttributeError("Size mismatch")

    cdef object call(self, CLCommandQueue queue):
        cdef cl_event event
        cdef cl_int errcode
        errcode = clEnqueueWriteBuffer(queue._command_queue,
                                      self._dst._mem,
                                      self._blocking,
                                      self._dst._offset,
                                      self._cb,
                                      self._src.data,
                                      0, NULL,
                                      &event)
        if errcode < 0: raise Exception("Excep")
        return _createCLEvent(event, queue)

cdef class CLUnmapBuffer(CLCommand):
    """
    This command enqueue the unmapping of a buffer from the device
    memory to the Host memory
    """
    def __cinit__(self, CLMappedBuffer buffer):
        self._dst = buffer

    cdef object call(self, CLCommandQueue queue):
        cdef cl_event event
        cdef cl_int errcode
        if not self._dst._ready: raise AttributeError("Already Unmap")
        errcode = clEnqueueUnmapMemObject(queue._command_queue,
                                          self._dst._buffer._mem, self._dst._address,
                                          0, NULL,
                                          &event)
        if errcode < 0: raise Exception("Excep")
        self._dst._ready = False
        return _createCLEvent(event, queue)

cdef class CLReadImageNDArray(CLCommand):
    def __cinit__(self, np.ndarray dst, CLImage src, cl_bool blocking = True):
            self._dst = dst
            self._src = src
            self._shape[1] = self._shape[2] = 1
            self._origin[0] = self._origin[1] = self._origin[2] = 0
            self._shape[0] = dst.shape[0]
            self._blocking = blocking
            self._slice_pitch = self._row_pitch = 0
            if dst.ndim > 1:
                self._shape[1] = dst.shape[1]
                self._row_pitch = dst.strides[0]
            if dst.ndim > 2:
                self._shape[2] = dst.shape[2]
                self._row_pitch = dst.strides[1]
                self._slice_pitch = dst.strides[0]

    cdef object call(self, CLCommandQueue queue):
        cdef cl_event event
        cdef cl_int errcode
        errcode = clEnqueueReadImage(queue._command_queue,
                                   self._src._mem,
                                   self._blocking,
                                   self._origin, self._shape,
                                   self._row_pitch,
                                   self._slice_pitch,
                                   self._dst.data,
                                   0, NULL,
                                   &event)
        if errcode < 0: raise translateError(errcode)
        return _createCLEvent(event, queue)

cdef class CLWriteImageNDArray(CLCommand):

    def __cinit__(self, CLImage dst, np.ndarray src, cl_bool blocking = True):
            self._dst = dst
            self._src = src
            self._shape[1] = self._shape[2] = 1
            self._origin[0] = self._origin[1] = self._origin[2] = 0
            self._shape[0] = src.shape[0]
            self._blocking = blocking
            self._slice_pitch = self._row_pitch = 0
            if src.ndim > 1:
                self._shape[1] = src.shape[1]
                self._row_pitch = src.strides[0]
            if src.ndim > 2:
                self._shape[2] = src.shape[2]
                self._row_pitch = src.strides[1]
                self._slice_pitch = src.strides[0]

    cdef object call(self, CLCommandQueue queue):
        cdef cl_event event
        cdef cl_int errcode
        errcode = clEnqueueWriteImage(queue._command_queue,
                                   self._dst._mem,
                                   self._blocking,
                                   self._origin, self._shape,
                                   self._row_pitch,
                                   self._slice_pitch,
                                   self._src.data,
                                   0, NULL,
                                   &event)
        if errcode < 0: raise translateError(errcode)
        return _createCLEvent(event, queue)

cdef class CLBarrier(CLCommand):
    """
    A synchronization point that enqueues a barrier operation.
    """
    cdef object call(self, CLCommandQueue queue):
        cdef cl_int errcode
        errcode = clEnqueueBarrier(queue._command_queue)
        if errcode < 0: raise translateError(errcode)

cdef class CLMarker(CLCommand):
    """
    Enqueues a marker command.
    """
    cdef object call(self, CLCommandQueue queue):
        cdef cl_event event
        cdef cl_int errcode
        errcode = clEnqueueMarker(queue._command_queue, &event)
        if errcode < 0: raise translateError(errcode)
        return _createCLEvent(event, queue)

cdef class CLMapBuffer(CLCommand):
    """
    This command map a Device Buffer to the Host memory buffer
    """
    def __cinit__(self, CLMappedBuffer dst, CLBuffer src, cl_map_flags flags, cl_bool blocking = True):
        self._flags = flags
        self._blocking = True
        self._src = src
        self._dst = dst

    cdef object call(self, CLCommandQueue queue):
        cdef cl_event event
        cdef cl_int errcode
        if self._dst._ready: raise AttributeError("Buffer Already mapped")
        self._dst._address = clEnqueueMapBuffer(queue._command_queue,
                                                self._src._mem,
                                                self._blocking,
                                                self._flags,
                                                self._src._offset, self._src.size - self._src._offset ,
                                                0, NULL, &event,
                                                &errcode)
        if errcode < 0: raise translateError(errcode)
        self._dst._buffer = self._src
        self._dst._ready = True
        return _createCLEvent(event, queue)
