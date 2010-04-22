======
======
Example usage::
    
    import cycl
    import numpy as np

    kernel = \
    """
    __kernel void fill(__global int *data, int value)
    {
        const int i = get_global_id(0);
        data[i] = value;
    }
    """

    platform = cycl.getPlatforms()[0]
    device = p.getDevices()[0]
    ctx = p.createContext([device])

    device_buffer = c.createBuffer(512 * 512 * 4)
    host_buffer = np.zeros((512, 512), 'int32')
    queue = ctx.createCommandQueue(device)

    program = ctx.createProgramWithSource(kernel).build()
    print program.getBuildLog(device)

    kernel = program.createKernel("fill")
    kernel.parameters = (cycl.parameter_type.MEM_TYPE, cycl.parameter_type.INT_TYPE)

    kernel.setArgs(b, 10)

    cmd1 = cycl.CLNDRangeKernel(kernel, global_work_size = (512 * 512, 1, 1), local_work_size = (256, 1, 1))
    cmd2 = cycl.CLReadBufferNDArray(host_buffer, device_buffer)

    queue.enqueue(cmd1)
    queue.enqueue(cmd2)

    queue.finish()
    print host_buffer
