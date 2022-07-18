import numpy as np
from calculation_apis import *
from Interval import print_2d_array

def can_convert_real_numbered_array_to_Interval_array():
    a = np.ones((3,4))
    b = to_interval_array_np(a)
    assert(b.dtype == object)
    assert(b.shape == a.shape)
    assert(b[0,0] == Interval(1,1))

def can_convert_Interval_array_to_upper():
    a = np.array([[Interval(1,1), Interval(1,2)], [Interval(1,3), Interval(1,4)]])
    b = get_upper(a)
    assert(b.dtype == np.float32)
    assert(b.shape == a.shape)
    assert(b[1,1] == 4)

def can_convert_Interval_array_to_lower():
    a = np.array([[Interval(1,1), Interval(1,2)], [Interval(1,3), Interval(1,4)]])
    b = get_lower(a)
    assert(b.dtype == np.float32)
    assert(b.shape == a.shape)
    assert(b[1,1] == 1)

def gemm_non_interval_cpu():
    A = np.ones((2,3))
    B = np.ones((3,2))
    bias = np.ones((2,2))
    dest = mat_mul(A, B, False, False)
    assert(dest[0][0] == 3)
    dest = gemm(A, B, bias,False, False)
    assert(dest[0][0] == 4)


def gemm_non_interval_gpu():
    A = np.ones((2,3))
    B = np.ones((3,2))
    bias = np.ones((2,2))
    dest = mat_mul(A, B, False, True)
    assert(dest[0][0] == 3)
    dest = gemm(A, B, bias,False, True)
    assert(dest[0][0] == 4)

def gemm_interval_cpu():
    A = to_interval_array_np(np.ones((2,3)))
    B = to_interval_array_np(np.ones((3,2)))
    bias = to_interval_array_np(np.ones((2,2)))
    dest = mat_mul(A, B, True, False)
    print(dest[0][0])
    dest = gemm(A, B, bias, True, False)
    print(dest[0][0])


# This test fails because A and B created by this way is actually an np array of objects
# and each has 48 bytes!
# if their data is passed directly, then C function will not work!
# either rewrite a new C func or recreate a data structure and pass to C function
def gemm_interval_gpu():
    A = to_interval_array_np(np.ones((2,3)))
    B = to_interval_array_np(np.ones((3,2)))
    bias = to_interval_array_np(np.ones((2,2)))
    ll = ctypes.cdll.LoadLibrary
    lib = ll('./gemmc/gemmGPU.so')
    dest = mat_mul(A, B, True, True, lib)
    print(dest[0][0])
    dest = gemm(A, B, bias, True, True, lib)
    print(dest[0][0])


if __name__ == '__main__':
    can_convert_real_numbered_array_to_Interval_array()
    can_convert_Interval_array_to_upper()
    can_convert_Interval_array_to_lower()
    gemm_non_interval_cpu()
    gemm_non_interval_gpu()
    gemm_interval_cpu()
    gemm_interval_gpu()
