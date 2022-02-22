#!/usr/bin/env python3

import sys
sys.path.append('../../..')

import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt


# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
  scales = [0.0,0.01,0.05,0.1,0.15]
  
  plt.style.use('single')
  for scale in scales:
    data = xr.load_dataset("scale-%3.2f.nc" % scale)

    strain = data.strains.data[:,0]
    stress = data.stresses.data[:,0]

    plt.plot(strain, stress)
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.tight_layout()
    plt.title("Scale = %3.2f" % scale)
    plt.show()
