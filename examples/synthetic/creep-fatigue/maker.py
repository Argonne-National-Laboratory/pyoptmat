#!/usr/bin/env python3

import sys
sys.path.append('../../..')

import numpy as np
import numpy.random as ra

import pandas as pd
import xarray as xr

import os.path

import torch
from pyoptmat import models, flowrules, hardening, optimize, experiments
from pyoptmat.temperature import ConstantParameter as CP

import itertools

from tqdm import tqdm

import matplotlib.pyplot as plt

import tqdm

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
device = torch.device("cpu")

# Actual parameters
E_true = 150000.0
R_true = 200.0
d_true = 5.0
n_true = 7.0
eta_true = 300.0
s0_true = 50.0
C_true = [10000.0, 5000.0, 500.0]
g_true = [200.0, 150.0, 5.0]

# Scale factor used in the model definition 
sf = 0.5

def make_model(E, n, eta, s0, R, d, C, g, device = device, **kwargs):
  """
    Key function for the entire problem: given parameters generate the model
  """
  isotropic = hardening.VoceIsotropicHardeningModel(
      CP(R, scaling = optimize.bounded_scale_function((torch.tensor(R_true*(1-sf), device = device), torch.tensor(R_true*(1+sf), device = device)))),
      CP(d, scaling = optimize.bounded_scale_function((torch.tensor(d_true*(1-sf), device = device), torch.tensor(d_true*(1+sf), device = device))))) 
  kinematic = hardening.ChabocheHardeningModel(
      CP(C, scaling = optimize.bounded_scale_function((torch.tensor(C_true, device = device)*(1-sf), torch.tensor(C_true, device = device)*(1.0+sf)))), 
      CP(g, scaling = optimize.bounded_scale_function((torch.tensor(g_true, device = device)*(1-sf), torch.tensor(g_true, device = device)*(1.0+sf))))) 
  flowrule = flowrules.IsoKinViscoplasticity(
      CP(n, scaling = optimize.bounded_scale_function((torch.tensor(n_true*(1-sf), device = device), torch.tensor(n_true*(1+sf), device = device)))),
      CP(eta, scaling = optimize.bounded_scale_function((torch.tensor(eta_true*(1-sf), device = device), torch.tensor(eta_true*(1+sf), device = device)))),
      CP(s0, scaling = optimize.bounded_scale_function((torch.tensor(s0_true*(1-sf), device = device), torch.tensor(s0_true*(1+sf), device = device)))),
      isotropic, kinematic)
  model = models.InelasticModel(CP(E, scaling = optimize.bounded_scale_function((torch.tensor(E_true*(1-sf), device = device), torch.tensor(E_true*(1+sf), device = device)))),
      flowrule, **kwargs)

  return model

def generate_input(strain_ranges, strain_rates, hold_times, N, nload, nhold, zero = 1e-6):
  """
    Generate the times and strains given the strain ranges, strain rates, and hold times
  """
  time = []
  strain = []
  for rng, rate, hold in zip(strain_ranges, strain_rates, hold_times):
    t, e = experiments.sample_cycle_normalized_times({'max_strain': rng/2.0, 'R': -1.0, 'strain_rate': rate, 'tension_hold': hold,
      'compression_hold': zero}, N, nload = nload, nhold = nhold)
    time.append(t)
    strain.append(e)

  return torch.tensor(time).T, torch.tensor(strain).T

def load_data(scale, nsamples, device = device):
  """
    Helper to load datafiles back in
  """
  expdata = xr.open_dataset(os.path.join(os.path.dirname(__file__), "scale-%3.2f.nc" % scale))
  ntime = expdata.dims['time']
  times = expdata['times'].data[:,:,:nsamples].reshape((ntime,-1))
  strains = torch.tensor(expdata['strains'].data[:,:,:nsamples].reshape((ntime,-1)), device = device)
  stresses = expdata['stresses'].data[:,:,:nsamples].reshape((ntime,-1))

  return torch.tensor(times, device = device), strains, torch.zeros_like(strains), torch.tensor(stresses, device = device)

if __name__ == "__main__":
  zero = 1e-6 # Avoid dt = 0

  # Running this script will regenerate the data
  nback = 3
  N = 50
  nload = 20
  nhold = 20
  erate = 1.0e-3
  strain_ranges = list(np.logspace(np.log10(0.002),np.log10(0.02), 5))
  hold_times = [1e-6, 60.0, 5*60.0, 60*60.0]
  nsamples = 20

  combs = list(itertools.product(strain_ranges, hold_times))
  strain_ranges = [c[0] for c in combs]
  hold_times = [c[1] for c in combs]

  nconds = len(strain_ranges)
  
  times, strains = generate_input(strain_ranges, [erate]*nconds, hold_times, N, nload, nhold)

  ntime = times.shape[0]
  nconds = times.shape[1]

  strain_ranges = torch.tensor(strain_ranges)
  hold_times = torch.tensor(hold_times)

  scales = [0.0, 0.01, 0.05, 0.1, 0.15]
  
  for scale in scales:
    print("Generating data for scale = %3.2f" % scale)
    full_times = torch.empty((ntime, nconds, nsamples))
    full_strains = torch.empty_like(full_times)
    full_stresses = torch.empty_like(full_times)
    full_ranges = torch.empty(full_times.shape[1:])
    full_rates = torch.empty_like(full_ranges)
    full_holds = torch.empty_like(full_ranges)

    for i in tqdm.tqdm(range(nsamples)):
      full_times[:,:,i] = times
      full_strains[:,:,i] = strains
      full_ranges[:,i] = strain_ranges
      full_rates[:,i] = erate
      full_holds[:,i] = hold_times

      # True values are 0.5 with our scaling so this is easy
      model = make_model(torch.tensor(0.5), 
          torch.tensor(ra.normal(0.5, scale)),
          torch.tensor(ra.normal(0.5, scale)), torch.tensor(ra.normal(0.5, scale)),
          torch.tensor(ra.normal(0.5, scale)), torch.tensor(ra.normal(0.5, scale)),
          torch.tensor(ra.normal(0.5, scale, size = (nback,))), torch.tensor(ra.normal(0.5, scale, size = (nback,))))
      with torch.no_grad():
        res = model.solve(times, strains)[:,:,0]
        full_stresses[:,:,i] = res

    ds = xr.Dataset(
        {
          "times": (["time", "trial", "repeat"], full_times.numpy()),
          "strains": (["time", "trial", "repeat"], full_strains.numpy()),
          "stresses": (["time", "trial", "repeat"], full_stresses.numpy()),
          "strain_ranges": (["trial", "repeat"], full_ranges.numpy()),
          "strain_rates": (["trial", "repeat"], full_rates.numpy()),
          "hold_times": (["trial", "repeat"], full_holds.numpy())
          }, attrs = {"scale": scale})

    ds.to_netcdf("scale-%3.2f.nc" % scale)
