
cdef class CLCommand:
    cdef object call(self, CLCommandQueue queue):
        raise AttributeError("Abstract Method")

cdef class CLCopyBuffer(CLCommand):
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
        cdef CLEvent instance = CLEvent.__new__(CLEvent)
        instance._queue = queue
        instance._event = event
        return instance

cdef class CLReadBufferNDArray(CLCommand):
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
        cdef CLEvent instance = CLEvent.__new__(CLEvent)
        instance._event = event
        instance._queue = queue
        return instance


cdef class CLNDRangeKernel(CLCommand):
    def __cinit__(self, CLKernel kernel, tuple global_work_size = (1,1,1), tuple local_work_size = (1,1,1)):
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
        cdef CLEvent instance = CLEvent.__new__(CLEvent)
        instance._event = event
        instance._queue = queue
        return instance




cdef class CLWriteBufferNDArray(CLCommand):
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
        cdef CLEvent instance = CLEvent.__new__(CLEvent)
        instance._event = event
        instance._queue = queue
        return instance


cdef class CLUnmapBuffer(CLCommand):
    cdef CLEvent call(self, CLCommandQueue queue):
        raise AttributeError("Abstract Method")

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
        cdef CLEvent instance = CLEvent.__new__(CLEvent)
        instance._event = event
        instance._queue = queue
        return instance

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
        cdef CLEvent instance = CLEvent.__new__(CLEvent)
        instance._event = event
        instance._queue = queue
        return instance


cdef class CLBarrier(CLCommand):
    cdef object call(self, CLCommandQueue queue):
        cdef cl_int errcode
        errcode = clEnqueueBarrier(queue._command_queue)
        if errcode < 0: raise translateError(errcode)


cdef class CLMarker(CLCommand):
    cdef object call(self, CLCommandQueue queue):
        cdef cl_event event
        cdef cl_int errcode
        errcode = clEnqueueMarker(queue._command_queue, &event)
        if errcode < 0: raise translateError(errcode)
        cdef CLEvent instance = CLEvent.__new__(CLEvent)
        instance._queue = queue
        instance._event = event
        return instance

cdef class CLMapBuffer(CLCommand):
    def __cinit__(self, CLBuffer src, cl_map_flags flags, cl_bool blocking = True):
        self._size = src.size - src._offset
        self._flags = flags
        self._blocking = True
        self._src = src
        self._offset = src._offset

    cdef object call(self, CLCommandQueue queue):
        cdef void *address
        cdef cl_event event
        cdef cl_int errcode
        address = clEnqueueMapBuffer(queue._command_queue,
                                     self._src._mem,
                                     self._blocking,
                                     self._flags,
                                     self._offset, self._size,
                                     0, NULL, &event,
                                     &errcode)
        if errcode < 0: raise translateError(errcode)
        cdef CLEvent evt = CLEvent.__new__(CLEvent)
        cdef CLMappedBuffer instance = CLMappedBuffer.__new__(CLMappedBuffer)
        evt._event = event
        evt._queue = queue
        instance._command_queue = queue
        instance._address = address
        instance._buffer = self._src
        return evt, instance
