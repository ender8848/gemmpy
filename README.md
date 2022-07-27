# gemmpy

gemm Python API for sound floating-point number calculation

# environment setup

gemmpy contains gemmc C API and supports GPU acceleration for custom Interval type based on cutlass. To setup GPU environment, please refer to the [gemmc documentation](https://github.com/ender8848/cuda-playground/tree/main/gemmc#readme) in ```gemmc``` folder. 

# compile gemmc API into dynamic linking code

Use the following command to create a new lib folder and compile dynamic linking code

```
cp -r gemmc/src gemmc/lib
mv gemmc/lib/gemmGPU.cuh gemmc/lib/gemmGPU.cu
nvcc -O3 -shared -Xcompiler -fPIC gemmc/lib/gemmGPU.cu -o /gemmGPU.so
mv gemmc/lib/mma.cuh gemmc/lib/mma.cu
nvcc -O3 -shared -Xcompiler -fPIC gemmc/lib/mma.cu -o mma.so
```

# run the test

```
python calculation_apis_test.py
```