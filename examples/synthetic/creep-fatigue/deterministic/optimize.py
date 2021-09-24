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
def make(n, eta, s0, R, d, C, g, **kwargs):
  return make_model(torch.tensor(0.5), n, eta, s0, R, d, C, g, 
      device = device, use_adjoint = True,**kwargs).to(device)

if __name__ == "__main__":
  # 1) Load the data for the variance of interest,
  #    cut down to some number of samples, and flatten
  scale = 0.05
  nsamples = 15 # at each condition
  times, strains, true_stresses = load_data(scale, nsamples,
      device = device)

  # 2) Setup names for each parameter and the initial conditions
  names = ["n", "eta", "s0", "R", "d", "C", "g"]
  ics = [ra.uniform(0,1) for i in range(len(names[:-2]))]
  ics += [ra.uniform(0,1,size=(3,)), ra.uniform(0,1,size=3)]

  print("Initial parameter values:")
  for n, ic in zip(names[:-2], ics):
    print("%s:\t%3.2f" % (n, ic))
  print("%s:\t[%3.2f, %3.2f, %3.2f]" % ((names[-2],) + tuple(ics[-2])))
  print("%s:\t[%3.2f, %3.2f, %3.2f]" % ((names[-1],) + tuple(ics[-1])))
  print("")

  # 3) Create the actual model
  model = optimize.DeterministicModel(make, names, ics).to(device)

  # 4) Setup the optimizer
  niter = 10
  optim = torch.optim.LBFGS(model.parameters())

  # 5) Setup the objective function
  loss = torch.nn.MSELoss(reduction = 'sum')

  # 6) Actually do the optimization!
  def closure():
    optim.zero_grad()
    pred = model(times, strains)
    lossv = loss(pred, true_stresses)
    lossv.backward()
    return lossv

  t = tqdm(range(niter), total = niter, desc = "Loss:    ")
  loss_history = []
  for i in t:
    closs = optim.step(closure)
    loss_history.append(closs.detach().cpu().numpy())
    t.set_description("Loss: %3.2e" % loss_history[-1])
  
  # 7) Check accuracy of the optimized parameters
  print("")
  print("Optimized parameter accuracy:")
  for n in names[:-2]:
    print("%s:\t%3.2f/0.50" % (n, getattr(model, n).data))
  print("%s:\t[%3.2f, %3.2f, %3.2f]" % ((names[-2],) + tuple(getattr(model, 
    names[-2]).data)))
  print("%s:\t[%3.2f, %3.2f, %3.2f]" % ((names[-1],) + tuple(getattr(model, 
    names[-1]).data)))
  print("")

  # 8) Save the convergence history
  np.savetxt("loss-history.txt", loss_history)

  plt.figure()
  plt.plot(loss_history)
  plt.xlabel("Iteration")
  plt.ylabel("Loss")
  plt.tight_layout()
  plt.savefig("convergence.pdf")
