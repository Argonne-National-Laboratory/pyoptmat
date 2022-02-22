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
from pyoptmat.temperature import ConstantParameter as CP

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

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def make_model(E, n, eta, s0, R, d, device = torch.device("cpu"), **kwargs):
  """
    Key function for the entire problem: given parameters generate the model
  """
  isotropic = hardening.VoceIsotropicHardeningModel(
      CP(R, scaling = optimize.bounded_scale_function((torch.tensor(R_true*(1-sf), device = device), torch.tensor(R_true*(1+sf), device = device)))),
      CP(d, scaling = optimize.bounded_scale_function((torch.tensor(d_true*(1-sf), device = device), torch.tensor(d_true*(1+sf), device = device))))) 
  kinematic = hardening.NoKinematicHardeningModel()
  flowrule = flowrules.IsoKinViscoplasticity(
      CP(n, scaling = optimize.bounded_scale_function((torch.tensor(n_true*(1-sf), device = device), torch.tensor(n_true*(1+sf), device = device)))),
      CP(eta, scaling = optimize.bounded_scale_function((torch.tensor(eta_true*(1-sf), device = device), torch.tensor(eta_true*(1+sf), device = device)))),
      CP(s0, scaling = optimize.bounded_scale_function((torch.tensor(s0_true*(1-sf), device = device), torch.tensor(s0_true*(1+sf), device = device)))),
      isotropic, kinematic)
  model = models.InelasticModel(CP(E, scaling = optimize.bounded_scale_function((torch.tensor(E_true*(1-sf), device = device), torch.tensor(E_true*(1+sf), device = device)))),
      flowrule)

  return models.ModelIntegrator(model, **kwargs)

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
  strains = torch.tensor(expdata['strains'].data[:,:,:nsamples].reshape((ntime,-1)), device = device)
  stresses = expdata['stresses'].data[:,:,:nsamples].reshape((ntime,-1))

  return torch.tensor(times, device = device), strains, torch.zeros_like(strains, device = device), torch.tensor(stresses, device = device)

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
    full_temperatures = torch.zeros_like(full_strains)

    for i in tqdm.tqdm(range(nsamples)):
      full_times[:,:,i] = times
      full_strains[:,:,i] = strains

      # True values are 0.5 with our scaling so this is easy
      model = make_model(torch.tensor(0.5), 
          torch.tensor(ra.normal(0.5, scale)),
          torch.tensor(ra.normal(0.5, scale)), torch.tensor(ra.normal(0.5, scale)),
          torch.tensor(ra.normal(0.5, scale)), torch.tensor(ra.normal(0.5, scale)))

      with torch.no_grad():
        full_stresses[:,:,i] = model.solve_strain(times, strains, 
            full_temperatures[:,:,i])[:,:,0]
    
    full_cycles = torch.zeros_like(full_times, dtype = int)
    types = np.array(["tensile"] * (nsamples * len(erates)))
    controls = np.array(["strain"] * (nsamples * len(erates)))
    
    ds = xr.Dataset(
        {
          "time": (["ntime", "nexp"], full_times.flatten(-2,-1).numpy()),
          "strain": (["ntime", "nexp"], full_strains.flatten(-2,-1).numpy()),
          "stress": (["ntime", "nexp"], full_stresses.flatten(-2,-1).numpy()),
          "temperature": (["ntime", "nexp"], full_temperatures.flatten(-2,-1).numpy()),
          "cycle": (["ntime", "nexp"], full_cycles.flatten(-2,-1).numpy()),
          "types": (["nexp"], types),
          "control": (["nexp"], controls)
          }, attrs = {"scale": scale, "nrates": nrates, "nsamples": nsamples})

    ds.to_netcdf("scale-%3.2f.nc" % scale)
