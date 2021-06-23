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

import pyro
from pyro.infer import SVI, Trace_ELBO
import pyro.optim as optim

import matplotlib.pyplot as plt

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
dev = "cpu"
device = torch.device(dev)

# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, **kwargs):
  return make_model(torch.tensor(0.5), n, eta, s0, R, d, device = device,
      use_adjoint = True, **kwargs).to(device)

if __name__ == "__main__":
  # 1) Load the data for the variance of interest,
  #    cut down to some number of samples, and flatten
  scale = 0.05
  nsamples = 25 # at each strain rate
  times, strains, true_stresses = load_data(scale, nsamples, device = device)

  # 2) Setup names for each parameter and the priors
  names = ["n", "eta", "s0", "R", "d"]
  loc_loc_priors = [torch.tensor(ra.uniform(0.25,0.75), device = device) for i in range(len(names))]
  loc_scale_priors = [torch.tensor(0.10, device = device) for i in range(len(names))]
  scale_scale_priors = [torch.tensor(0.10, device = device) for i in range(len(names))]

  eps = torch.tensor(1.0e-4, device = device)

  print("Initial parameter values:")
  print("\tloc loc\t\tloc scale\tscale scale")
  for n, llp, lsp, sp in zip(names, loc_loc_priors, loc_scale_priors, 
      scale_scale_priors):
    print("%s:\t%3.2f\t\t%3.2f\t\t%3.2f" % (n, llp, lsp, sp))
  print("")

  # 3) Create the actual model
  model = optimize.HierarchicalStatisticalModel(make, names, loc_loc_priors,
      loc_scale_priors, scale_scale_priors, eps).to(device)

  # 4) Get the guide
  guide = model.make_guide()
  
  # 5) Setup the optimizer and loss
  lr = 5.0e-3
  niter = 4000
  num_samples = 1
  
  optimizer = optim.Adam({"lr": lr})
  svi = SVI(model, guide, optimizer, 
      loss = Trace_ELBO(num_particles = num_samples))

  # 6) Infer!
  t = tqdm(range(niter), total = niter, desc = "Loss:    ")
  loss_hist = []
  for i in t:
    loss = svi.step(times, strains, true_stresses)
    loss_hist.append(loss)
    t.set_description("Loss %3.2e" % loss)

  # 7) Print out results
  print("")
  print("Inferred distributions:")
  print("\tloc\t\tscale")
  for n in names:
    print("%s:\t%3.2f/0.50\t%3.2f/%3.2f" % (n,
      pyro.param("AutoDelta." + n + model.loc_suffix).data,
      pyro.param("AutoDelta." + n + model.scale_suffix).data,
      scale))
  print("")

  # 8) Save some info
  np.savetxt("loss-history.txt", loss_hist)

  plt.figure()
  plt.plot(loss_hist)
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.tight_layout()
  plt.savefig("convergence.pdf")
