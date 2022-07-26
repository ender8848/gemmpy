# cuda-playground
not yet up-to-date

# Prerequisite

|required|version|
|--------|-------|
|GPU hardware||
|PyTorch|compatible with CUDA|
|CUDA|compatible with PyTorch|

A brief summary of CUDA and PyTorch installation

1. check CUDA version on device using the following command

```
nvidia-smi
```

or

```
nvcc -V
```

If both works, remember the version of CUDA

If only one of those works, consider adding some path to the environmental variable. 

If neither works, then you may have to reinstall the CUDA driver (see steps later)

2. Get PyTorch with specific version in this [link](https://pytorch.org/get-started/previous-versions/)

If CUDA version matches one of the PyTorch requirements, then simply install PyTorch and skip step3, otherwise go to step3 and install CUDA. 

3. CUDA installation

If either CUDA is not installed or has an unmatched version, then go to this [link](https://developer.nvidia.com/cuda-toolkit-archive) to install CUDA. 

# test cases

## CPU and GPU matrix addition test

CPU matrix addition is done in PyTorch and GPU matrix addition is done by calling function in dynamic linking library.

1. Compile .cu file into .so file using nvcc

```
nvcc -O3 -shared -Xcompiler -fPIC sum_matrix.cu -o sum_matrix.so
```

2. execute the python script 

Execute with specified size n, the script will perform a nXn matrix addition on CPU and GPU. Default size is 1024x1024

```
python MMASpeedTest.py [-o size]
```

---
just a backup

```
nvcc -O3 -shared -Xcompiler -fPIC gemmGPU.cu -o gemmGPU.so
```