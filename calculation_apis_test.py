import numpy as np
import torch
from calculation_apis import *
from Interval import print_2d_array


def can_convert_real_numbered_np_array_to_Interval_np_array():
    """
    test function np_array_float2interval
    """
    a = np.array([[1,2],[3,4]], dtype = np.float32)
    b = np_array_float2interval(a)
    assert(b.dtype == object)
    assert(b.shape == a.shape)
    assert(b[0,0] == Interval(1,1))
    assert(b[0,1] == Interval(2,2))
    assert(b[1,0] == Interval(3,3))
    assert(b[1,1] == Interval(4,4))


def can_convert_float_torch_array_to_pesudo_interval_array():
    """
    test function torch_array_float2pinterval
    """
    # gpu case
    a = torch.tensor([[1,2],[3,4]], dtype = torch.float32, device='cuda')
    b = torch_array_float2pinterval(a)
    assert(b.dtype == a.dtype)
    assert(b.device == a.device)
    assert(torch.equal(b, torch.tensor([[1,1,2,2],[3,3,4,4]], dtype = torch.float32, device = 'cuda')))


def can_convert_numpy_Interval_array_to_upper():
    """
    test function get_upper for numpy interval array
    """
    # numpy interval case
    a = np.array([[Interval(1,1), Interval(1,2)], [Interval(1,3), Interval(1,4)]])
    b = get_upper(a)
    assert(b.dtype == np.float32)
    assert(np.array_equal(b,np.array([[1,2],[3,4]], dtype = np.float32)))
    

def can_convert_numpy_Interval_array_to_lower():
    """
    test function get_lower for numpy interval array
    """
    # numpy interval case
    a = np.array([[Interval(1,1), Interval(1,2)], [Interval(1,3), Interval(1,4)]])
    b = get_lower(a)
    assert(b.dtype == np.float32)
    assert(np.array_equal(b,np.array([[1,1],[1,1]], dtype = np.float32)))


def can_convert_torch_pseudo_interval_array_to_upper():
    """
    test function get_upper for torch pseudo interval array
    """
    # torch pseudo interval case
    a = torch.tensor([[1,2,3,4], [5,6,7,8]], dtype = torch.float32, device='cuda')
    b = get_upper(a)
    assert(b.dtype == a.dtype)
    assert(b.device == a.device)
    assert(torch.equal(b,torch.tensor([[2,4],[6,8]], dtype = torch.float32, device='cuda')))


def can_convert_torch_pseudo_interval_array_to_lower():
    """
    test function get_lower for torch pseudo interval array
    """
    # torch pseudo interval case
    a = torch.tensor([[1,2,3,4], [5,6,7,8]], dtype = torch.float32, device='cuda')
    b = get_upper(a)
    assert(b.dtype == a.dtype)
    assert(b.device == a.device)
    assert(torch.equal(b,torch.tensor([[1,3],[5,7]], dtype = torch.float32, device='cuda')))


def add_test():
    # numpy float case
    A = np.array([[1,2],[3,4],[5,6]])
    B = np.ones((2,3))
    res = add(A, B)
    assert(res)





    # torch case
    

def gemm_non_interval_gpu():
    A = np.ones((2,3))
    B = np.ones((3,2))
    bias = np.ones((2,2))
    dest = mat_mul(A, B, False, True)
    assert(dest[0][0] == 3)
    dest = gemm(A, B, bias,False, True)
    assert(dest[0][0] == 4)

def gemm_interval_cpu():
    A = np_array_float2interval(np.ones((2,3)))
    B = np_array_float2interval(np.ones((3,2)))
    bias = np_array_float2interval(np.ones((2,2)))
    dest = mat_mul(A, B, True, False)
    print(dest[0][0])
    dest = gemm(A, B, bias, True, False)
    print(dest[0][0])


# This test fails because A and B created by this way is actually an np array of objects
# and each has 48 bytes!
# if their data is passed directly, then C function will not work!
# either rewrite a new C func or recreate a data structure and pass to C function
# # But it seems not teh reason as float size is 24 ...
def gemm_interval_gpu():
    A = torch_array_float2pinterval(np.ones((2,3)))
    B = torch_array_float2pinterval(np.ones((3,2)))
    bias = torch_array_float2pinterval(np.ones((2,2)))
    ll = ctypes.cdll.LoadLibrary
    lib = ll('./gemmc/gemmGPU.so')
    dest = mat_mul(A, B, True, True, lib)
    dest = np_array_pinterval2interval(dest)
    print(dest[0][0])
    dest = gemm(A, B, bias, True, True, lib)
    dest = np_array_pinterval2interval(dest)
    print(dest[0][0])


if __name__ == '__main__':
    can_convert_real_numbered_np_array_to_Interval_np_array()
    can_convert_float_torch_array_to_pesudo_interval_array()
    can_convert_numpy_Interval_array_to_upper()
    can_convert_numpy_Interval_array_to_lower()
    can_convert_torch_pseudo_interval_array_to_upper()