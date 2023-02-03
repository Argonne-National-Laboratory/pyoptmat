#!/usr/bin/env python

import numpy as np 

import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("results.txt", 'r') as f:
        names = f.readline().strip().split(" ")
        data = np.loadtxt(f)
    
    for i, label in enumerate(names):
        plt.plot(data[0], data[i+1], label = label)

    plt.legend(loc='best')
    plt.show()

    speedup = data[1,0] / data[2]
    plt.plot(data[0], speedup)
    plt.xlabel("Chunk size")
    plt.ylabel("Speedup")
    plt.show()

