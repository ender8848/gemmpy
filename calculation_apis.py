import numpy as np
from Interval import Interval
def to_interval_array_np(arr):
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

def get_upper(arr):
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


def get_lower(arr):
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


def add(A, B):
    """
    perform matrix addition
    """
    pass

def mat_mul(A, B):
    """
    perform matrix multiplication
    """
    pass

def gemm(A, B, C):
    """
    performs gemm
    """
    pass
