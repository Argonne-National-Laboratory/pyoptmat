#!/usr/bin/env python3

import sys
sys.path.append('../../..')
sys.path.append('.')

import numpy as np
import numpy.random as ra

import xarray as xr
import torch

from maker import make_model, load_data, sf

from pyoptmat import optimize
from tqdm import tqdm

import matplotlib.pyplot as plt

import time
import warnings
warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Run on GPU!
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
# Run on CPU (home machine GPU is eh)
# On that machine 2 threads is optimal
dev = "cpu"
device = torch.device(dev)

# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, **kwargs):
  return make_model(torch.tensor(0.5), n, eta, s0, R, d, use_adjoint = True,
      device = device, **kwargs).to(device)

if __name__ == "__main__":
  # 1) Load the data for the variance of interest,
  #    cut down to some number of samples, and flatten
  scale = 0.05
  nsamples = 20 # at each strain rate
  times, strains, true_stresses = load_data(scale, nsamples, device = device)

  # 2) Setup names for each parameter and the initial conditions
  names = ["n", "eta", "s0", "R", "d"]
  ics = [ra.uniform(0,1) for i in range(len(names))]

  # 3) Create the actual model
  model = optimize.DeterministicModel(make, names, ics)

  # 4) Run some number of times
  loss = torch.nn.MSELoss(reduction = 'sum')
  niter = 2
  t1 = time.time()
  t = tqdm(range(niter), total = niter)
  for i in t:
    model.zero_grad()
    pred = model(times, strains)
    lossv = loss(pred, true_stresses)
    lossv.backward()

  te = time.time() - t1

  print("Elapsed time: %f s" % te)
