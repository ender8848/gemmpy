import numpy as np
from Interval import Interval
import cupy as cp
import ctypes

def np_array_float2interval(arr:np.ndarray):
    """
    converts an real-valued array to an interval array
    args: 
        arr: array-like
    returns:
        an numpy array converting from arr
    
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be numpy array")
    result = np.empty((arr.shape),dtype=Interval)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i,j] = Interval(arr[i,j], arr[i,j])
    return result

def array_float2pinterval(arr):
    """
    converts an real-valued array to an peudo interval array which uses continuous memory view to mimic an Interval
    args: 
        arr: array-like, can be numpy array or cupy array
    returns:
        an pseudo interval np or cp float array converting from arr
    """
    if isinstance(arr, np.ndarray):
        result = np.zeros((arr.shape[0], arr.shape[1]*2),dtype=np.float32)
    elif isinstance(arr, cp.ndarray):
        result = cp.zeros((arr.shape[0], arr.shape[1]*2),dtype=cp.float32)
    else:
        raise TypeError("arr must be numpy array or cupy array")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            result[i,2*j] = arr[i][j]
            result[i,2*j+1] = arr[i][j]
    return result

def np_array_pinterval2interval(arr):
    """
    converts an peudo interval array to an interval array
    args: 
        arr: array-like pesudo interval
    returns:
        an interval numpy array converting from arr
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("arr must be numpy array")
    result = np.empty((arr.shape[0], arr.shape[1]//2),dtype=Interval)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]//2):
            result[i,j] = Interval(arr[i,2*j], arr[i,2*j+1])
    return result

def get_upper(arr:np.ndarray):
    """
    converts an interval numpy array or pseudo interval array 
    to corresponding upper-value float array
    args: 
        arr: Interval array or pseudo interval array
    returns:
        upper value float array
    
    """
    if isinstance(arr, np.ndarray) and arr.dtype != np.float32:
        result = np.empty((arr.shape), dtype = np.float32)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result[i,j] = arr[i,j].upper
        return result
    if isinstance(arr, np.ndarray):
        result = np.empty((arr.shape[0], arr.shape[1]//2), dtype = np.float32)
    elif isinstance(arr, cp.ndarray):
        result = cp.empty((arr.shape[0], arr.shape[1]//2), dtype = cp.float32)
    else:
        raise TypeError("arr must be numpy array or cupy array")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]//2):
            result[i,j] = arr[i,2*j+1]
    return result


def get_lower(arr:np.ndarray):
    """
    converts an interval numpy array or pseudo interval array 
    to corresponding lower-value float array
    args: 
        arr: Interval array or pseudo interval array
    returns:
        lower value float array
    
    """
    if isinstance(arr, np.ndarray) and arr.dtype != np.float32:
        result = np.empty((arr.shape), dtype = np.float32)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                result[i,j] = arr[i,j].lower
        return result
    if isinstance(arr, np.ndarray):
        result = np.empty((arr.shape[0], arr.shape[1]//2), dtype = np.float32)
    elif isinstance(arr, cp.ndarray):
        result = cp.empty((arr.shape[0], arr.shape[1]//2), dtype = cp.float32)
    else:
        raise TypeError("arr must be numpy array or cupy array")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]//2):
            result[i,j] = arr[i,2*j]
    return result
    


# not completely implemented, currently only compatible with CPU numpy array
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

def mat_mul(A:np.ndarray, B:np.ndarray, interval = True, gpu = True, lib = None):
    """
    perform matrix multiplication A @ B
    args:
        A: lhs of matrix multiplication, use 2 floats to represent an interval, NOT AN INTERVAL ARRAY, but float array because it has continuous memory view
        B: rhs of matrix multiplication, use 2 floats to represent an interval, NOT AN INTERVAL ARRAY, but float array because it has continuous memory view
        interval: whether to use sound calculation
        gpu: whether to run on GPU
        lib: dynamic linking lib
    returns:
        new np array A @ B, use 2 floats to represent an interval, NOT AN INTERVAL ARRAY, but float array because it has continuous memory view
    """
    if not gpu:
        return A @ B
    if not interval:
        # use cupy
        A_ = cp.array(A)
        B_ = cp.array(B)
        return cp.asnumpy(A @ B)
    # interval case
    dest = array_float2pinterval(np.zeros((A.shape[0], int(B.shape[1]/2))))
    gemmGPUPy(A, B, dest, A.shape[0], int(B.shape[1]/2), int(A.shape[1]/2), lib)
    return dest
    

def gemm(A:np.ndarray, B:np.ndarray, bias:np.ndarray, interval = True, gpu = True, lib = None):
    """
    performs gemm dest = A @ B + bias and return dest
    args:
        A: lhs of matrix multiplication, use 2 floats to represent an interval, NOT AN INTERVAL ARRAY, but float array because it has continuous memory view
        B: rhs of matrix multiplication, use 2 floats to represent an interval, NOT AN INTERVAL ARRAY, but float array because it has continuous memory view
        bias: bias matrix 
        interval: whether to use sound calculation
        gpu: whether to run on GPU
        lib: dynamic linking lib
    returns:
        dest of matrix multiplication, use 2 floats to represent an interval, NOT AN INTERVAL ARRAY, but float array because it has continuous memory view
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
    dest = array_float2pinterval(np.zeros((A.shape[0], int(B.shape[1]/2))))
    gemmGPUPy(A, B, dest, A.shape[0], int(B.shape[1]/2), int(A.shape[1]/2), lib, bias)
    return dest