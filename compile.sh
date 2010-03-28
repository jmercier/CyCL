FILES="opencl"

for file in ${FILES}; 
do
    echo ${file}
    mako-render ${file}.pyx.mako > ${file}.pyx
    cython ${file}.pyx
    wc -l ${file}.c
done

gcc opencl.c -shared -fPIC -o opencl.so -I/usr/local/cuda/include -I/usr/include/python2.6 -lpython2.6  -O3 -L/usr/local/stream/lib  -lOpenCL -I/usr/local/stream/include
