from functools import wraps

import string
import cycl
import math

import numpy as np

import logging
log = logging.getLogger(__name__)

def memoize(fct):
    """
    This is a decorator which cache the result of a function based on the 
    given parameter.
    """
    return_dict = {}

    @wraps(fct)
    def wrapper(*args):
        if args not in return_dict:
            return_dict[args] = fct(*args)
        return return_dict[args]
    return wrapper

type_defines = """
#define float32 float
#define int32 int
"""


__axpy_template__ = string.Template("""
__kernel void axpy(__global ${ftype} *a,
                   ${ftype} x,
                   __global ${ftype} *y,
                   __const int size)
{
    int index = get_global_id(0);
    if (index < size)
    {
        y[index] += a[index] * x;
    }
}
""")

__dot_template__ = string.Template("""
__kernel void dot_product(__global float *result,
                     __global ${ftype} *v1,
                     __global ${ftype} *v2,
                     __const int size)
{
    __local float buffer[1024];
    int local_index = get_local_id(0);
    buffer[local_index] = 0;

    for (int i = get_global_id(0); i < size; i += get_global_size(0))
    {
        buffer[local_index] += v1[i] * v2[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = get_local_size(0) / 2; i > 0; i = i / 2)
    {
        if (local_index < i)
        {
            buffer[local_index] = buffer[local_index + i] + buffer[local_index];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0)
    {
        result[get_group_id(0)] = buffer[local_index];
    }
}


""")

__spmv_template__ = string.Template("""
__kernel void csr_spmv(__global ${ftype}    *vout,
                       __global ${ftype}    *data,
                       __global int         *indices,
                       __global int         *indptr,
                       __global ${ftype}    *vinp,
                       int                  size)
{
    int pos = get_global_id(0);
    if (pos < size)
    {

        float result = 0;

        for (int i = indptr[pos]; i < indptr[pos + 1]; i++)
        {
            result += vinp[pos] * data[indices[i]];
        }

        vout[pos] = result;
    }
}
""")



def __compile_program__(ctx, program_text):
    program = ctx.createProgramWithSource(type_defines + program_text)
    try:
        program.build()
    except cycl.CLError, e:
        if log.isEnabledFor(logging.WARNING):
            for d in ctx._devices:
                print program.getBuildLog(d)
            raise
    return program

@memoize
def __get_axpy_kernel__(ctx, ftype):
    program = __compile_program__(ctx, __axpy_template__.substitute(ftype = ftype))
    kernel = program.createKernel("axpy")
    kernel.parameters = (cycl.parameter_type.MEM_TYPE,
                         cycl.parameter_type.string_dict[ftype],
                         cycl.parameter_type.MEM_TYPE,
                         cycl.parameter_type.INT_TYPE)

    return kernel


@memoize
def __get_dot_kernel__(ctx, ftype):
    program = __compile_program__(ctx, __dot_template__.substitute(ftype = ftype))
    kernel = program.createKernel("dot_product")
    kernel.parameters = (cycl.parameter_type.MEM_TYPE,
                         cycl.parameter_type.MEM_TYPE,
                         cycl.parameter_type.MEM_TYPE,
                         cycl.parameter_type.INT_TYPE)

    return kernel

@memoize
def __get_spmv_kernel__(ctx, ftype):
    program = __compile_program__(ctx, __spmv_template__.substitute(ftype = ftype))
    kernel = program.createKernel("csr_spmv")
    kernel.parameters = (cycl.parameter_type.MEM_TYPE,
                         cycl.parameter_type.MEM_TYPE,
                         cycl.parameter_type.MEM_TYPE,
                         cycl.parameter_type.MEM_TYPE,
                         cycl.parameter_type.MEM_TYPE,
                         cycl.parameter_type.INT_TYPE)
    return kernel

class CLCSRMatrix(object):
    def __init__(self, context, spmatrix):
        matrix = spmatrix.tocsr()

        self.data       = context.createBufferLike(matrix.data)
        self.indices    = context.createBufferLike(matrix.indices)
        self.indptr     = context.createBufferLike(matrix.indptr)

        self.shape      = matrix.shape
        self.dtype      = spmatrix.dtype

        self._context = context

    def send(self, spmatrix, **kw):
        send_data = cycl.CLWriteBufferNDArray(self.data,
                                              spmatrix.data,
                                              **kw)
        send_indices = cycl.CLWriteBufferNDArray(self.indices,
                                                 spmatrix.indices,
                                                 **kw)
        send_intptr = cycl.CLWriteBufferNDArray(self.indptr,
                                                spmatrix.indptr,
                                                blocking = blocking)

        return [send_data, send_indices, send_indptr]

def elementwise(kernel, size, device):
    lws = 2.0 ** 8
    if device is not None:
        lws = self.kernel.getWorkGroupSize(device)
    gws = math.ceil(size / lws) * lws
    return [cycl.CLNDRangeKernel(kernel,
                                 global_work_size = (gws, 1, 1),
                                 local_work_size = (lws, 1, 1))]


def axpy(a, y, x = 1, device = None):
    size = a.size / a.dtype.itemsize
    kernel = __get_axpy_kernel__(a._context, str(a.dtype))
    kernel.setArgs(a, x, y, size)
    return elementwise(kernel, size, device)

def dot(r, v1, v2, device = None):
    size = v1.size / v1.dtype.itemsize
    kernel = __get_dot_kernel__(v1._context, str(v1.dtype))
    kernel.setArgs(r, v1, v2, size)
    return elementwise(kernel, size, device)

def spvm(csrmat, out, inp, device = None):
    size = out.size / out.dtype.itemsize
    if csrmat.shape[0] != size:
        raise ValueError("Matrix-Vector Alignment Mismatch : Incompatible Output")
    if csrmat.shape[1] * out.dtype.itemsize != inp.size:
        raise ValueError("Matrix-Vector Alignment Mismatch : Incompatible Input")

    kernel = __get_spmv_kernel__(out._context, str(out.dtype))
    kernel.setArgs(out, csrmat.data, csrmat.indices, csrmat.indptr, inp, size)

    return elementwise(kernel, size, device)

if __name__ == '__main__':
    import numpy as np
    import cycl

    import scipy as sc
    import scipy.sparse

    import eikonal.cllinear as linear


    p = cycl.getPlatforms()[0]
    d = p.getDevices()[0]
    c = p.createContext([d])
    q = c.createCommandQueue(d)

    size = 64 * 64 * 64

    cpu_sp = sc.sparse.eye(size, size, dtype = 'float32').tocsr() * 4
    cpu_sp = sc.sparse.dia_matrix((([1] * size, [1] * size, [1] * size), (-1, 0, 1)), shape = (size, size), dtype = 'float32').tocsr()
    gpu_sp = linear.CLCSRMatrix(c, cpu_sp)
    q.enqueue(gpu_sp.send(cpu_sp))

    cpu_a = np.ones(size, dtype = 'float32')
    cpu_b = np.zeros(size, dtype = 'float32')

    gpu_a = c.createTypedBuffer(size, 'float32')
    gpu_b = c.createTypedBuffer(size, 'float32')

    writecmd    = [cycl.CLWriteBufferNDArray(gpu_a, cpu_a)]
    writespcmd  = gpu_sp.send(cpu_sp)

    spmvcmd     = spvm(gpu_sp, gpu_b, gpu_a)

    readbcmd     = [cycl.CLReadBufferNDArray(cpu_b, gpu_b)]

    dotcmd      =  dot(gpu_b, gpu_b, gpu_a)
    readacmd     = [cycl.CLReadBufferNDArray(cpu_a, gpu_a)]



    q.enqueue(writecmd + writespcmd +  spmvcmd + readbcmd + dotcmd + readacmd)
    q.finish()



