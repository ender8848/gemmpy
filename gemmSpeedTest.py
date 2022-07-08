import argparse
import ctypes
import time
import torch
import numpy as np

# parse cmd input args
parser = argparse.ArgumentParser()
parser.add_argument('-s', dest='size', help="both dimension size in matrix")
args = parser.parse_args()

def gemmGPUPy(A_dev, B_dev, dest_dev, M, N, K, bias_dev = None):
    """
    make sure that dest_dev is 0 if there is no bias
    In this case dest_dev will be used as bias
    """
    M_c = ctypes.c_int(M)
    N_c = ctypes.c_int(N)
    K_c = ctypes.c_int(K)
    datatype = ctypes.c_int(0) # namely float
    a_p = ctypes.cast(A_dev.data_ptr(), ctypes.c_void_p)
    b_p = ctypes.cast(B_dev.data_ptr(), ctypes.c_void_p)
    dest_p = ctypes.cast(dest_dev.data_ptr(), ctypes.c_void_p)
    if bias_dev is None:
        bias_c = ctypes.cast(dest_dev.data_ptr(), ctypes.c_void_p)
    else:
        bias_c = ctypes.cast(bias_dev.data_ptr(), ctypes.c_void_p)
    
    lib.gemmGPUPy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.gemmGPUPy(a_p, b_p, dest_p, M_c, N_c, K_c, datatype)


if __name__ == '__main__':

    # some basic settings, must init device in python
    torch.cuda.init()
    cuda0 = torch.device('cuda:0')
    if (args.size is None):
        M = 1024;
        N = 1024;
        K = 1024;
    else: 
        M = int(args.size)
        N = int(args.size)
        K = int(args.size)
    print(f"Comparing matrix multiplication with size: [{M},{K}] & [{K},{N}]")

    # initialize tensors in CPU
    A_host = torch.ones(M, K, dtype=torch.float32, device='cpu')
    B_host = torch.ones(K, N, dtype=torch.float32, device='cpu')
    C_host = torch.zeros(M, N, dtype=torch.float32, device='cpu')

    # test CPU matrix addition time -- 100 loop
    start = time.time()
    for i in range (1):
        C_host = A_host @ B_host
    duration = time.time() - start
    print(f"CPU matrix multiplication 100 times in PyTorch costs {duration:.6f} seconds")

    # initialize tensors in GPU
    A_dev = torch.ones(M, K, dtype=torch.float, device=cuda0)
    B_dev = torch.ones(K, N, dtype=torch.float, device=cuda0)
    C_dev = torch.zeros(M, N, dtype=torch.float, device=cuda0)
    
    # test GPU matrix addition time
    start = time.time()
    ll = ctypes.cdll.LoadLibrary        
    lib = ll('./gemmGPU.so')
    gemmGPUPy(A_dev, B_dev, C_dev, M, N, K)
    duration = time.time() - start
    print(f"GPU matrix multiplication 100 times in calling C costs {duration:.6f} seconds")
    # copy c_dev from gpu to c_host in cpu

    
    # free GPU memory 
    # not really needed coz it is PyTorch's concern to do memory management, just trust PyTorch
    torch.cuda.reset_accumulated_memory_stats() 