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

def generate_input(stresses, stressrates, max_time, ntime, fload):
  """
    Generate the times and strains given the strain rates, maximum strain, and number of time steps
  """
  times = torch.ones_like(stresses) * max_time
  nload = int(ntime * fload)
  nhold = ntime - nload
  temps = torch.zeros_like(stresses)
  
  time, stresses, temperatures, cyclces = experiments.make_creep_tests(stresses, temps, stressrates, times,
      nload, nhold)

  return time, stresses, temperatures

def load_data(scale, nsamples, device = torch.device("cpu")):
  """
    Helper to load datafiles back in
  """
  expdata = xr.open_dataset(os.path.join(os.path.dirname(__file__), "scale-%3.2f.nc" % scale))
  ntime = expdata.dims['time']
  times = expdata['times'].data[:,:,:nsamples].reshape((ntime,-1))
  strains = torch.tensor(expdata['strains'].data[:,:,:nsamples].reshape((ntime,-1)), device = device)
  stresses = expdata['stresses'].data[:,:,:nsamples].reshape((ntime,-1))
  temperatures = torch.tensor(expdata['temperatures'].data[:,:,:nsamples].reshape((ntime,-1)), device = device)

  return torch.tensor(times, device = device), strains, temperatures, torch.tensor(stresses, device = device)

if __name__ == "__main__":
  # Running this script will regenerate the data
  ntime = 200
  load_frac = 0.2
  ntests = 50
  nsamples = 50

  stresses = torch.linspace(100,200,ntests)
  srates = torch.logspace(0.01,1,ntests)

  max_time = 5000*3600.0

  scales = [0.0, 0.01, 0.05, 0.1, 0.15]

  times, stresses, temperatures = generate_input(stresses, srates, max_time, ntime, load_frac)

  for scale in scales:
    print("Generating data for scale = %3.2f" % scale)
    full_times = torch.empty((ntime, ntests, nsamples))
    full_strains = torch.empty_like(full_times)
    full_temperatures = torch.empty_like(full_times)
    full_stresses = torch.empty_like(full_times)

    for i in tqdm.tqdm(range(nsamples)):
      full_times[:,:,i] = times
      full_stresses[:,:,i] = stresses
      full_temperatures[:,i] = temperatures

      # True values are 0.5 with our scaling so this is easy
      model = make_model(torch.tensor(0.5), 
          torch.tensor(ra.normal(0.5, scale)),
          torch.tensor(ra.normal(0.5, scale)), torch.tensor(ra.normal(0.5, scale)),
          torch.tensor(ra.normal(0.5, scale)), torch.tensor(ra.normal(0.5, scale)))
      inter = models.ModelIntegrator(model)
      with torch.no_grad():
        full_strains[:,:,i] =  model.solve_stress(times, stresses, temperatures)[:,:,0]

    ds = xr.Dataset(
        {
          "times": (["time", "trial", "repeat"], full_times.numpy()),
          "strains": (["time", "trial", "repeat"], full_strains.numpy()),
          "stresses": (["time", "trial", "repeat"], full_stresses.numpy()),
          "temperatures": (["time", "trial", "repeat"], full_temperatures.numpy())
          }, attrs = {"scale": scale})

    ds.to_netcdf("scale-%3.2f.nc" % scale)
