#!/usr/bin/env python3

import sys
sys.path.append('../..')

import numpy as np
import torch

import matplotlib.pyplot as plt

import torch.autograd.profiler as profiler

from pyoptmat import models, flowrules, hardening, experiments
from pyoptmat.temperature import ConstantParameter as CP

import time

if __name__ == "__main__":
  torch.set_default_tensor_type(torch.DoubleTensor)
  # Run on GPU!
  if torch.cuda.is_available():
      dev = "cuda:0"
  else:
      dev = "cpu"
  dev = "cpu"
  device = torch.device(dev)

  E = torch.tensor(100000.0, device = device)
  n = torch.tensor(5.2, device = device)
  eta = torch.tensor(500.0, device = device)
  R = torch.tensor(100.0, device = device)
  d = torch.tensor(5.1, device = device)
  C = torch.tensor(1000.0, device = device)
  g = torch.tensor(10.0, device = device)
  s0 = torch.tensor(10.0, device = device)

  nbatch = 50
  nload = 20
  nhold = 20
  N = 20
  substeps = 2

  ntime = (4*nload + 2*nhold) * N + 1

  times = np.zeros((ntime,nbatch))
  strains = np.zeros((ntime,nbatch))

  for i in range(nbatch):
    cycle = experiments.generate_random_cycle(R = [-1,-0.99])
    times[:,i], strains[:,i] = experiments.sample_cycle_normalized_times(cycle, N, 
        nload = nload, nhold = nhold)
  
  times = torch.tensor(times, device = device)
  strains = torch.tensor(strains, device = device)
  temperatures = torch.zeros_like(strains)

  iso = hardening.VoceIsotropicHardeningModel(CP(R), CP(d))
  kin = hardening.FAKinematicHardeningModel(CP(C), CP(g))

  model = models.ModelIntegrator(models.InelasticModel(CP(E),
      flowrules.IsoKinViscoplasticity(CP(n), CP(eta),
        CP(s0), iso, kin)), substeps = substeps).to(device)
  
  ts = time.time()
  with torch.no_grad():
    res = model.solve_strain(times, strains, temperatures)
  tt = time.time() - ts

  print("Total time: %f" % tt)
  print("Efficiency: %f steps/s" % ((ntime * substeps * nbatch)/tt))
