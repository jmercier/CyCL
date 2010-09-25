import cycl
import numpy as np

kernel = \
"""
__kernel void t1(__global int *data, int value)
{
    const int i = get_global_id(0);
    data[i] = value;
}
"""

p = cycl.getPlatforms()[0]
d = p.getDevices()[0]
c = p.createContext([d])

b = c.createBuffer(512 * 512 * 4)
bhost = np.zeros((512, 512), 'int32')
q = c.createCommandQueue(d)

p = c.createProgramWithSource(kernel).build()

print p.getBuildLog(d)

k1 = p.createKernel("t1")
k1.parameters = (cycl.parameter_type.MEM_TYPE, cycl.parameter_type.INT_TYPE)
k1.setArgs(b, 10)

cmd = cycl.CLNDRangeKernel(k1, global_work_size = ( 512 * 512, 1, 1), local_work_size = (256, 1, 1))
cmd2 = cycl.CLReadBufferNDArray(bhost, b)

q.enqueue([cmd, cmd2])
q.finish()
print bhost

