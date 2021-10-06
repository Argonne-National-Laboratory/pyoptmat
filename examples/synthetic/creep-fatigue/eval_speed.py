#!/usr/bin/env python3

import sys
sys.path.append('../../../..')
sys.path.append('..')

import numpy as np
import numpy.random as ra

import xarray as xr
import torch

from maker import make_model, load_data, sf

from pyoptmat import optimize
from tqdm import tqdm

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import time

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Run on GPU!
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

# Run on CPU for running multiple jobs on workstation
# Optimal number of threads is about 4 on our machine
dev = "cpu"
device = torch.device(dev)

# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, C, g, **kwargs):
  return make_model(torch.tensor(0.5), n, eta, s0, R, d, C, g,
      device = device, use_adjoint = True, **kwargs).to(device)

if __name__ == "__main__":
  # 1) Load the data for the variance of interest,
  #    cut down to some number of samples, and flatten
  scale = 0.05
  nsamples = 20 # at each condition
  times, strains, temperatures, true_stresses = load_data(scale, nsamples, device = device)

  sf = 0.1
  use = int(len(times)*sf)

  # Move to device and curtail to some number of steps
  times = times[:use]
  strains = strains[:use]
  temperatures = temperatures[:use]
  true_stresses = true_stresses[:use]

  # 2) Setup names for each parameter and the initial conditions
  names = ["n", "eta", "s0", "R", "d", "C", "g"]
  ics = [ra.uniform(0,1) for i in range(len(names[:-2]))]
  ics += [ra.uniform(0,1,size=(3,)), ra.uniform(0,1,size=3)]

  # 3) Create the actual model
  model = optimize.DeterministicModel(make, names, ics).to(device)

  # 4)  Run some number of times
  loss = torch.nn.MSELoss(reduction = 'sum')
  t1 = time.time()
  niter = 2
  t = tqdm(range(niter), total = niter)
  for i in t:
    model.zero_grad()
    pred = model(times, strains, temperatures)
    lossv = loss(pred, true_stresses)
    lossv.backward()
  
  te = time.time() - t1

  print("Total run time: %f s" % te)
