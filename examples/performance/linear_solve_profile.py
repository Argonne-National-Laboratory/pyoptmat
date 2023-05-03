#!/usr/bin/env python3

import time
import random

import matplotlib.pyplot as plt

import numpy as np
import torch
import pandas as pd

from pyoptmat import chunktime

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def run_time(operator, D, L, v, repeat = 1):
    times = []
    for i in range(repeat):
        t1 = time.time()
        op = operator(D.clone(), L.clone())
        x = op(v.clone())
        times.append(time.time() - t1)

    return np.mean(times)

if __name__ == "__main__":
    # Number of repeated trials to average over
    avg = 3
    # Size of the blocks in the matrix
    nsize = 10
    # Batch size: number of matrices to solve at once
    nbat = 10
    # Maximum number of blocks in the matrix
    max_blk = 1000
    # Number of samples in range(1,max_blk) to sample
    num_samples = 10
    
    nblks = sorted(random.sample(list(range(1,max_blk)), num_samples))

    methods = [chunktime.BidiagonalThomasFactorization, chunktime.BidiagonalPCRFactorization,
            lambda A, B: chunktime.BidiagonalHybridFactorization(A, B, min_size = 8),
            lambda A, B: chunktime.BidiagonalHybridFactorization(A, B, min_size = 16),
            lambda A, B: chunktime.BidiagonalHybridFactorization(A, B, min_size = 32),
            lambda A, B: chunktime.BidiagonalHybridFactorization(A, B, min_size = 64),
            lambda A, B: chunktime.BidiagonalHybridFactorization(A, B, min_size = 128)]

    method_names = ["Thomas", "PCR", "Hybrid, n = 8", "Hybrid, n = 16", "Hybrid, n = 32", 
            "Hybrid, n = 64", "Hybrid, n = 128"]

    nmethods = len(methods)
    ncase = len(nblks)

    times = np.zeros((nmethods, ncase))
    
    # Do this once to warm up the GPU, it seems to matter
    run_time(methods[0], torch.rand(3, nbat, nsize, nsize, device = device),
            torch.rand(2, nbat, nsize, nsize, device = device) / 10.0,
            torch.rand(nbat, 3 * nsize, device = device))

    for i,nblk in enumerate(nblks):
        print(nblk)
        D = torch.rand(nblk, nbat, nsize, nsize, device = device)
        L = torch.rand(nblk - 1, nbat, nsize, nsize, device = device) / 10.0

        v = torch.rand(nbat, nblk * nsize, device = device) 
        for j, method in enumerate(methods):
            times[j,i] = run_time(method, D, L, v, repeat = avg)

    data = pd.DataFrame(data = times.T, index = nblks, columns = method_names)
    data.avg = avg
    data.nsize = nsize
    data.nbat = nbat
    
    data.to_csv(f"{nbat}_{nsize}.csv")
