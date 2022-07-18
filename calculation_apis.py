import numpy as np
from Interval import Interval
import cupy as cp
import ctypes

def to_interval_array_np(arr:np.ndarray):
    """
    converts an real-valued array to an interval array
    args: 
        arr: array-like
    returns:
        an numpy array converting from arr
    
    """
    result = np.empty((arr.shape),dtype=Interval)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i,j] = Interval(arr[i,j], arr[i,j])
    return result

def get_upper(arr:np.ndarray):
    """
    converts an interval array to corresponding upper-value float array
    args: 
        arr: Interval array
    returns:
        upper value float array
    
    """
    result = np.empty((arr.shape), dtype = np.float32)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i,j] = arr[i,j].upper
    return result


def get_lower(arr:np.ndarray):
    """
    converts an interval array to corresponding lower-value float array
    args: 
        arr: Interval array
    returns:
        lower value float array
    
    """
    result = np.empty((arr.shape), dtype = np.float32)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i,j] = arr[i,j].lower
    return result


def add(A:np.ndarray, B:np.ndarray):
    """
    perform matrix addition, addition defaults to CPU implementation because it is relatively cheap to compute
    Calling C API may result in longer time
    Now that it is CPU implementation, Interval and non-Interval operations can all be done by + operator
    args:
        A: input array
        B: output array
    returns:
        new array A + B
    """
    return A + B

def gemmGPUPy(A_host:np.ndarray, B_host:np.ndarray, dest_host:np.ndarray, M, N, K, lib, bias_host:np.ndarray = None, is_host = True):
    """
    make sure that dest_dev is 0 if there is no bias
    In this case dest_dev will be used as bias
    """
    M_c = ctypes.c_int(M)
    N_c = ctypes.c_int(N)
    K_c = ctypes.c_int(K)
    datatype = ctypes.c_int(2) # namely interval float
    a_p = ctypes.cast(A_host.__array_interface__['data'][0], ctypes.c_void_p)
    b_p = ctypes.cast(B_host.__array_interface__['data'][0], ctypes.c_void_p)
    is_host_c = ctypes.c_bool(is_host)
    dest_p = ctypes.cast(dest_host.__array_interface__['data'][0], ctypes.c_void_p)
    if bias_host is None:
        # create a nullptr
        bias_c = ctypes.c_void_p(0)
    else:
        bias_c = ctypes.cast(bias_host.__array_interface__['data'][0], ctypes.c_void_p)
    
    lib.gemmGPUPy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_bool]
    lib.gemmGPUPy(a_p, b_p, dest_p, M_c, N_c, K_c, datatype, bias_c, is_host_c)
    print()

def mat_mul(A:np.ndarray, B:np.ndarray, interval = True, gpu = True, lib = None):
    """
    perform matrix multiplication A @ B
    args:
        A: lhs of matrix multiplication
        B: rhs of matrix multiplication
        interval: whether to use sound calculation
        gpu: whether to run on GPU
        lib: dynamic linking lib
    returns:
        new np array A @ B
    """
    if not gpu:
        return A @ B
    if not interval:
        # use cupy
        A_ = cp.array(A)
        B_ = cp.array(B)
        return cp.asnumpy(A @ B)
    # interval case
    dest = to_interval_array_np(np.zeros((A.shape[0], B.shape[1])))
    gemmGPUPy(A, B, dest, A.shape[0], B.shape[1], A.shape[1], lib)
    return dest
    

def gemm(A:np.ndarray, B:np.ndarray, bias:np.ndarray, interval = True, gpu = True, lib = None):
    """
    performs gemm dest = A @ B + bias and return dest
    args:
        A: lhs of matrix multiplication
        B: rhs of matrix multiplication
        bias: bias matrix 
        interval: whether to use sound calculation
        gpu: whether to run on GPU
        lib: dynamic linking lib
    returns:
        dest of matrix multiplication
    """

    if bias is None:
        return mat_mul(A, B, interval, gpu, lib)
    if not gpu:
        return A @ B + bias
    if not interval:
        # use cupy
        A_ = cp.array(A)
        B_ = cp.array(B)
        return cp.asnumpy(A @ B) + bias
    # interval case
    dest = np.empty((A.shape[0], B.shape[1]), dtype = Interval)
    gemmGPUPy(A, B, dest, A.shape[0], B.shape[1], A.shape[1], lib)
    return dest
