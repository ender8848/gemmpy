//
// Created by hao on 07/07/22.
//

#ifndef CUTLASSTEST_GEMMGPU_CUH
#define CUTLASSTEST_GEMMGPU_CUH
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
// #include "cudastart.cuh"
#include "Interval.cuh"


enum datatype {
    FLOAT = 0,
    DOUBLE = 1,
    INTV_FLOAT = 2,
    INTV_DOUBLE = 3
};

template<typename T>
void gemmGPUC(T* A_dev, T* B_dev, T* dest_dev, int M, int N, int K, T* bias_dev = nullptr) {
    using Gemm = cutlass::gemm::device::Gemm<
            T,                           // ElementA
            cutlass::layout::ColumnMajor,              // LayoutA
            T,                           // ElementB
            cutlass::layout::ColumnMajor,              // LayoutB
            T,                           // ElementOutput
            cutlass::layout::ColumnMajor,              // LayoutOutput
            T,                                     // ElementAccumulator
            cutlass::arch::OpClassSimt,            // tag indicating Tensor Cores
            cutlass::arch::Sm61              // tag indicating target GPU compute architecture (61 for GTX 1060)
    >;
    Gemm gemm_op;
    cutlass::Status status;

    // Define alpha and beta, use 1 for gemm in matrix multiplication
    T alpha = T(1.);
    T beta = T(1.);
    int lda = M;
    int ldb = K;
    int ldc = M;
    int ldd = M;
    //
    // Launch GEMM on the device
    //
    status = gemm_op({
        {M, N, K},
        {A_dev, lda},            // TensorRef to A device tensor
        {B_dev, ldb},            // TensorRef to B device tensor
        {bias_dev, ldc},            // TensorRef to C device tensor
        {dest_dev, ldd},            // TensorRef to D device tensor - may be the same as C (depending on passed value)
        {alpha, beta}           // epilogue operation arguments
    });

    if (status != cutlass::Status::kSuccess) {
        printf("GEMM failed\n");
        printf(cutlassGetStatusString(status));
        printf("\n");
    }
}



extern "C" {
// perform float gemm D = AB+C
void gemmGPUPy(void* A_dev, void* B_dev, void* dest_dev, int M, int N, int K, int datatype, void* bias_dev = nullptr) {
    switch (datatype) {
        case datatype::FLOAT:
            gemmGPUC<float>(static_cast<float*>(A_dev),
                            static_cast<float*>(B_dev),
                            static_cast<float*>(dest_dev),
                            M, N, K,
                            static_cast<float*>(bias_dev));
            break;
        case datatype::DOUBLE:
            gemmGPUC<double>(static_cast<double*>(A_dev),
                             static_cast<double*>(B_dev),
                             static_cast<double*>(dest_dev),
                             M, N, K,
                             static_cast<double*>(bias_dev));
            break;
        case datatype::INTV_FLOAT:
            gemmGPUC<Interval<float>>(static_cast<Interval<float>*>(A_dev),
                                      static_cast<Interval<float>*>(B_dev),
                                      static_cast<Interval<float>*>(dest_dev),
                                      M, N, K,
                                      static_cast<Interval<float>*>(bias_dev));
            break;
        case datatype::INTV_DOUBLE:
            gemmGPUC<Interval<double>>(static_cast<Interval<double>*>(A_dev),
                                       static_cast<Interval<double>*>(B_dev),
                                       static_cast<Interval<double>*>(dest_dev),
                                       M, N, K,
                                       static_cast<Interval<double>*>(bias_dev));
            break;
        default:
            break;
    }
}
};



#endif //CUTLASSTEST_GEMMGPU_CUH
