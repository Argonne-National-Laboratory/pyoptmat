#!/usr/bin/env python3

import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

if __name__ == "__main__":
  scales = [0.0,0.01,0.05,0.1,0.15]

  erange = 0.02
  hold_time = 5*60.0
  
  for scale in scales:
    database = xr.open_dataset("scale-%3.2f.nc" % scale)
    print("database looks like: ", database)
    
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(database.strain[:, 0, :], database.stress[:, 0, :])
    plt.xlabel("Strain (mm/mm)", fontsize=16)
    plt.ylabel("Stress (MPa)", fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax = plt.gca()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2e'))
    # ax.xaxis.major.formatter._useMathText = True
    # ax.set_xticklabels(ax.get_xticks(), rotation = 10)
    
    plt.tight_layout()
    plt.savefig("cyclic-visualize-%3.2f.png" % scale, dpi=300)
    # plt.show()
    # plt.close()
