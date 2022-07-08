#include <cuda_runtime.h>
#include <stdio.h>
#include "cudastart.cuh"


// kernel function
// each thread calculates one item in matrix 
__global__ void sumMatrix2D(float * MatA,float * MatB,float * MatC,int nx,int ny)
{
    int ix = threadIdx.x+blockDim.x*blockIdx.x;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int idx = ix+iy*ny;
    if (ix<nx && iy<ny)
    {
        MatC[idx] = MatA[idx]+MatB[idx];
    }
}

extern "C" {
    void sumMatrix2DGPU(void* C_dev, void* A_dev, void* B_dev, int nx, int ny) {
        auto A_dev_ = static_cast<float*>(A_dev);
        auto B_dev_ = static_cast<float*>(B_dev);
        auto C_dev_ = static_cast<float*>(C_dev);
        double gpuStart = cpuSecond();
        // initDevice(0); // This should be caller's concern
        // 2-d bolck ，32×32
        dim3 block(32, 32);
        // 2-d grid，128×128
        dim3 grid((nx-1)/block.x+1, (ny-1)/block.y+1);


        //将核函数放在线程网格中执行
        sumMatrix2D<<<grid,block>>>(A_dev_, B_dev_, C_dev_, nx, ny);
        CHECK(cudaDeviceSynchronize());
        // cudaDeviceReset();
        double gpuTime = cpuSecond() - gpuStart;
        printf("GPU matrix addition in C costs: %f sec\n", gpuTime);
    }
}