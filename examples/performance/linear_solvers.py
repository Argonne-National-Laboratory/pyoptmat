#!/usr/bin/env python

import sys

sys.path.append("../..")

import time

from pyoptmat import solvers

import torch

import time

torch.set_default_tensor_type(torch.DoubleTensor)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

if __name__ == "__main__":
    nbatch = 10000
    nsize = 20

    A = torch.rand((nbatch,nsize,nsize), device = device)
    b = torch.rand((nbatch,nsize), device = device)
    
    t1 = time.time()
    for i in range(10):
        x = solvers.lu_gmres(A, b, check = min(nsize+1,20), maxiter = nsize+1)
    time_gmres = time.time() - t1
    
    t1 = time.time()
    for i in range(10):
        x = solvers.lu_linear_solve(A, b)
    time_lu = time.time() - t1

    print("GMRES: %f" % time_gmres)
    print("LU: %f" % time_lu)

