#!/usr/bin/env python3

"""
    Simple helper to make a plot illustrating the variation in the
    synthetic experimental data.
"""

import sys

sys.path.append("../../..")

import xarray as xr
import torch
import matplotlib.pyplot as plt


# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    scales = [0.0, 0.01, 0.05, 0.1, 0.15]

    for scale in scales:
        data = xr.load_dataset("scale-%3.2f.nc" % scale)

        strain = data.strain.data.reshape(-1, data.nrates, data.nsamples)
        stress = data.stress.data.reshape(-1, data.nrates, data.nsamples)
        
        for i in range(data.nrates):
            plt.plot(strain[:, i], stress[:, i])
            plt.xlabel("Strain (mm/mm)")
            plt.ylabel("Stress (MPa)")
            plt.title("Scale = %3.2f" % scale)
            plt.tight_layout()
            plt.show()
