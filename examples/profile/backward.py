#!/usr/bin/env python3

import sys
sys.path.append('../..')

import numpy as np
import torch

import matplotlib.pyplot as plt

import torch.autograd.profiler as profiler
from torch.nn import Parameter

from pyoptmat import models, flowrules, hardening, experiments

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

  nbatch = 50
  nload = 20
  nhold = 20
  substeps = 1

  nback = 3

  E = Parameter(torch.linspace(100000.0,11000.0,nbatch))
  n = Parameter(torch.linspace(5.2,5.6,nbatch))
  eta = Parameter(torch.linspace(110.0,115.0,nbatch))
  R = Parameter(torch.linspace(100.0,110.0,nbatch))
  d = Parameter(torch.linspace(5.1,5.6,nbatch))
  C = Parameter(torch.linspace(1000.0,1100.0,nbatch*nback).reshape(nbatch,nback))
  g = Parameter(torch.linspace(10.0,15.0,nbatch*nback).reshape(nbatch,nback))
  s0 = Parameter(torch.linspace(10.0,20.0,nbatch))

  N = 10

  model = models.InelasticModel(E, 
      flowrules.IsoKinViscoplasticity(n, eta, s0,
        hardening.VoceIsotropicHardeningModel(R, d),
        hardening.ChabocheHardeningModel(C, g)), use_adjoint = True,
      substeps = substeps).to(device)

  time1, strain1 = experiments.sample_cycle_normalized_times({'max_strain': 0.005,
    'R': -1.0, 'strain_rate': 1.0e-3, 'tension_hold': 600.0,
    'compression_hold': 1.0e-3}, N, nload = nload, nhold = nhold)
  times = torch.zeros(len(time1), nbatch)
  strains = torch.zeros(len(time1), nbatch)
  for i in range(nbatch):
    times[:,i] = torch.tensor(time1)
    strains[:,i] = torch.tensor(strain1)
  
  t1 = time.time()
  res = torch.norm(model.solve(times, strains))
  res.backward()
  tt = time.time() - t1

  ntime = times.shape[0] * substeps

  print("Total time: %f s" % tt)
  print("Efficiency: %f steps/s" % (ntime * nbatch / tt))


