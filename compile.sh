FILES="opencl"

for file in ${FILES}; 
do
    echo ${file}
    sed -e 's/\(.*\)\(CL_SAFE_CALL \)\(.*$\)/\1cdef cl_int res = \3\n\1if res < 0: raise translateError(res)/' \
	-e 's/\(.*\)\(CL_SAFE_CALL_NO_INIT \)\(.*$\)/\1res = \3\n\1if res < 0: raise translateError(res)/' \
	-e 's/\(.*\)\(CL_DEALLOC \)\(.*$\)/\1def __dealloc__(self):\n\1    cdef cl_int res = \3\n\1    if res < 0: print("Deallocation failure < %s >" % self.__class__.__name__)/' \
     ${file}.pyx.in > ${file}.pyx
    cython ${file}.pyx
    wc -l ${file}.c
done

gcc opencl.c -shared -fPIC -o opencl.so -I/usr/local/cuda/include -I/usr/include/python2.6 -lpython2.6  -O3 -L/usr/local/stream/lib  -latiocl64 -I/usr/local/stream/include
