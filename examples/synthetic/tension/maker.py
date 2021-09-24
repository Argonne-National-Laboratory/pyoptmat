#!/usr/bin/env python3

import sys
sys.path.append('../../..')

import numpy as np
import numpy.random as ra

import pandas as pd
import xarray as xr

import os.path

import torch
from pyoptmat import models, flowrules, hardening, optimize

from tqdm import tqdm

import matplotlib.pyplot as plt

import tqdm

import warnings
warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Actual parameters
E_true = 150000.0
R_true = 200.0
d_true = 5.0
n_true = 7.0
eta_true = 300.0
s0_true = 50.0

# Scale factor used in the model definition 
sf = 0.5

def make_model(E, n, eta, s0, R, d, device = torch.device("cpu"), **kwargs):
  """
    Key function for the entire problem: given parameters generate the model
  """
  isotropic = hardening.VoceIsotropicHardeningModel(
      R, d, 
      R_scale = optimize.bounded_scale_function((torch.tensor(R_true*(1-sf), device = device), torch.tensor(R_true*(1+sf), device = device))),
      d_scale = optimize.bounded_scale_function((torch.tensor(d_true*(1-sf), device = device), torch.tensor(d_true*(1+sf), device = device))))
  kinematic = hardening.NoKinematicHardeningModel()
  flowrule = flowrules.IsoKinViscoplasticity(
      n, eta, s0,
      isotropic, kinematic,
      n_scale = optimize.bounded_scale_function((torch.tensor(n_true*(1-sf), device = device), torch.tensor(n_true*(1+sf), device = device))),
      eta_scale = optimize.bounded_scale_function((torch.tensor(eta_true*(1-sf), device = device), torch.tensor(eta_true*(1+sf), device = device))),
      s0_scale = optimize.bounded_scale_function((torch.tensor(s0_true*(1-sf), device = device), torch.tensor(s0_true*(1+sf), device = device))))
  model = models.InelasticModel(E, flowrule, 
      E_scale = optimize.bounded_scale_function((torch.tensor(E_true*(1-sf), device = device), torch.tensor(E_true*(1+sf), device = device))),
      **kwargs)

  return model

def generate_input(erates, emax, ntime):
  """
    Generate the times and strains given the strain rates, maximum strain, and number of time steps
  """
  strain = torch.repeat_interleave(torch.linspace(0, emax, ntime)[None,:], len(erates), 0).T.to(device)
  time = strain / erates

  return time, strain

def load_data(scale, nsamples, device = torch.device("cpu")):
  """
    Helper to load datafiles back in
  """
  expdata = xr.open_dataset(os.path.join(os.path.dirname(__file__), "scale-%3.2f.nc" % scale))
  ntime = expdata.dims['time']
  times = expdata['times'].data[:,:,:nsamples].reshape((ntime,-1))
  strains = expdata['strains'].data[:,:,:nsamples].reshape((ntime,-1))
  stresses = expdata['stresses'].data[:,:,:nsamples].reshape((ntime,-1))

  return torch.tensor(times, device = device), torch.tensor(strains, device = device), torch.tensor(stresses, device = device)

if __name__ == "__main__":
  # Running this script will regenerate the data
  ntime = 200
  emax = 0.5
  erates = torch.logspace(-2,-8,7)
  nrates = len(erates)
  nsamples = 50

  scales = [0.0, 0.01, 0.05, 0.1, 0.15]

  times, strains = generate_input(erates, emax, ntime)

  for scale in scales:
    print("Generating data for scale = %3.2f" % scale)
    full_times = torch.empty((ntime, nrates, nsamples))
    full_strains = torch.empty_like(full_times)
    full_stresses = torch.empty_like(full_times)
    full_erates = torch.empty((nrates, nsamples))

    for i in tqdm.tqdm(range(nsamples)):
      full_times[:,:,i] = times
      full_strains[:,:,i] = strains
      full_erates[:,i] = erates

      # True values are 0.5 with our scaling so this is easy
      model = make_model(torch.tensor(0.5), 
          torch.tensor(ra.normal(0.5, scale)),
          torch.tensor(ra.normal(0.5, scale)), torch.tensor(ra.normal(0.5, scale)),
          torch.tensor(ra.normal(0.5, scale)), torch.tensor(ra.normal(0.5, scale)))
      with torch.no_grad():
        full_stresses[:,:,i] = model.solve(times, strains)[:,:,0]


    ds = xr.Dataset(
        {
          "times": (["time", "trial", "repeat"], full_times.numpy()),
          "strains": (["time", "trial", "repeat"], full_strains.numpy()),
          "stresses": (["time", "trial", "repeat"], full_stresses.numpy()),
          "rates": (["trial", "repeat"], full_erates.numpy())
          }, attrs = {"scale": scale})

    ds.to_netcdf("scale-%3.2f.nc" % scale)
