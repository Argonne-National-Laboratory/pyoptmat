#!/usr/bin/env python3

import time

import matplotlib.pyplot as plt

import torch

from pyoptmat import chunktime

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def times(nblk, nsize, nbat):
    D_blk = torch.rand(nblk, nbat, nsize, nsize, device = device)
    L_blk = torch.rand(nblk - 1, nbat, nsize, nsize, device = device) / 10.0

    b = torch.rand(nbat, nblk * nsize, device = device) 

    t = time.time()
    A = chunktime.BidiagonalPCRFactorization(D_blk, L_blk)    
    x = A(b)
    t1 = time.time() - t

    t = time.time()
    A = chunktime.BidiagonalThomasFactorization(D_blk, L_blk)    
    x = A(b)
    t2 = time.time() - t

    return t1, t2

if __name__ == "__main__":
    avg = 5
    nsize = 30
    nbat = 100
    
    #nblks = [10**i for i in range(0,4)]
    nblks = list(range(1,1000))
    time_pcr = []
    time_thomas = []

    pcr, thoms = times(2, nsize, nbat)

    for nblk in nblks:
        pcr = 0
        thomas = 0
        for i in range(avg):
            pcri, thomasi = times(nblk, nsize, nbat)
            pcr += pcri
            thomas += thomasi
        pcr /= avg
        thomas /= avg
        time_pcr.append(pcr)
        time_thomas.append(thomas)

    plt.plot(nblks, time_pcr, label = "PCR")
    plt.plot(nblks, time_thomas, label = "Thomas")
    plt.legend(loc = 'best')
    plt.xlabel("Chunk size")
    plt.ylabel("Wall time (s)")
    plt.show()
