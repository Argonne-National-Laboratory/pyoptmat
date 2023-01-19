#!/usr/bin/env python3

"""
    Tutorial example of training a statistical model to tension test data
    from from a known distribution.
"""

import sys
import os.path

sys.path.append("../../../..")
sys.path.append("..")

import numpy.random as ra

import xarray as xr
import torch

from maker import make_model, downsample, nsize

from pyoptmat import optimize, experiments
from tqdm import tqdm

import pyro
from pyro.infer import SVI
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
dev = "cpu"
device = torch.device(dev)

# Don't try to optimize for the Young's modulus
def make(*args, **kwargs):
    """
        Maker with Young's modulus fixed
    """
    return make_model(torch.tensor(0.5), *args, device=device, **kwargs).to(
        device
    )


if __name__ == "__main__":
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.05
    nsamples = 10  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )

    # 2) Setup names for each parameter and the priors
    names = ["n", "eta", "s0", "weights1", "biases1", "weights2", "biases2", "weights3", "biases3"]
    loc_loc_priors = [torch.tensor(ra.uniform(0, 1), device = device) for i in range(3)] + [
            torch.rand(nsize,1, device = device),
            torch.rand(nsize, device = device),
            torch.rand(nsize, nsize, device = device),
            torch.rand(nsize, device = device),
            torch.rand(1, nsize, device = device),
            torch.rand(1, device = device)]
    loc_scale_priors = [torch.ones_like(l)*0.15 for l in loc_loc_priors]
    scale_scale_priors = [torch.ones_like(l)*0.15 for l in loc_loc_priors]

    eps = torch.tensor(1.0e-4, device=device)

    # 3) Create the actual model
    model = optimize.HierarchicalStatisticalModel(
        make, names, loc_loc_priors, loc_scale_priors, scale_scale_priors, eps
    ).to(device)

    # 4) Get the guide
    guide = model.make_guide()

    # 5) Setup the optimizer and loss
    lr = 1.0e-3
    g = 1.0
    niter = 3500
    lrd = g ** (1.0 / niter)
    num_samples = 1

    optimizer = optim.ClippedAdam({"lr": lr, "lrd": lrd})

    ls = pyro.infer.Trace_ELBO(num_particles=num_samples)

    svi = SVI(model, guide, optimizer, loss=ls)

    # 6) Infer!
    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_hist = []
    for i in t:
        loss = svi.step(data, cycles, types, control, results)
        loss_hist.append(loss)
        t.set_description("Loss %3.2e" % loss)

    # Save parameters
    pyro.get_param_store().save("model.pyro")

    # 8) Plot convergence
    plt.figure()
    plt.loglog(loss_hist)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
