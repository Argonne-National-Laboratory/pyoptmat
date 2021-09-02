#!/usr/bin/env python3

import sys
sys.path.append('../../../..')
sys.path.append('..')

import numpy as np
import numpy.random as ra

import xarray as xr
import torch

from maker import make_model, load_data, sf

from pyoptmat import optimize, utility
from tqdm import tqdm

import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from infer import make

if __name__ == "__main__":
  # Results from a run using
  # scale = 0.05
  # nsamples = 25
  # eps = 1.0e-4

  # Trials
  ntrials = 25
  
  # True data
  scale = 0.05
  nsamples = 50 # Use all the data
  times, strains, true_stresses = load_data(scale, nsamples)
  
  # We can only easily plot one condition
  condition = 1 # Choice of condition
  times = times[:,condition*nsamples:(condition)*nsamples+1]
  strains = strains[:,condition*nsamples:(condition)*nsamples+1]
  true_stresses = true_stresses[:,condition*nsamples:(condition+1)*nsamples]
  
  # Actual results
  names = ["n", "eta", "s0", "R", "d"]
  loc = [0.51, 0.50, 0.49, 0.50, 0.50]
  scale = [0.02, 0.02, 0.02, 0.02, 0.02]

  eps = torch.tensor(1.0e-4)

  model = optimize.StatisticalModel(make, names, loc, scale, eps)

  stress_res = torch.empty(times.shape[0], ntrials)
  
  for i in range(ntrials):
    stress_res[:,i] = model.forward(times, strains)[:,0]

  utility.visualize_variance(strains[:,0], true_stresses, stress_res)
