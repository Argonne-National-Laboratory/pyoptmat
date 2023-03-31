#!/usr/bin/env python3

"""
    Just plot the data to make sure it seems reasonable
"""

import sys
sys.path.append('../../..')

import numpy as np
import torch
import xarray as xr
import matplotlib.pyplot as plt

from pyoptmat import experiments

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    input_data = xr.open_dataset("data.nc")
    data, results, cycles, types, control = experiments.load_results(
            input_data)

    plt.plot(data[-1].numpy(), results.numpy())
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.show()
