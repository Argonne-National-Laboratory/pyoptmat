#!/usr/bin/env python3

"""
    Example using the tutorial data to train a deterministic model, rather than
    a statistical model.
"""

import sys

sys.path.append("../../../..")
sys.path.append("..")

import os.path

import numpy.random as ra

import xarray as xr
import torch

from maker import make_model, downsample, params_true

from pyoptmat import optimize, experiments, scaling
from tqdm import tqdm

import matplotlib.pyplot as plt

# Don't care if integration fails
import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = "cpu"
device = torch.device(dev)

# Maker function returns the ODE model given the parameters
# Don't try to optimize for the Young's modulus
def make(g0, A, C, R, d, **kwargs):
    """
        Maker with the Young's modulus fixed
    """
    return make_model(g0, A, C, R, d, device=device, **kwargs).to(
        device
    )


if __name__ == "__main__":
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.0
    nsamples = 1  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )

    # 2) Setup names for each parameter and the initial conditions
    names = ["g0", "A", "C", "R", "d"]
    rng = 0.5
    ics = torch.tensor([ra.uniform((1-rng)*p,(1+rng)*p) for p in params_true])

    # 3) Calculate the initial gradient values
    model = optimize.DeterministicModel(lambda *args: make(*args), names, ics)
    loss = torch.nn.MSELoss(reduction="sum")

    lossv = loss(model(data, cycles, types, control), results)
    lossv.backward()
    print("Initial parameter gradients:")
    grads = []
    for n in names:
        grads.append(getattr(model,n).grad.detach())
        print("%s:\t%.3e" % (n, grads[-1]))
    print("")

    # 4) Set up some scaling functions
    scale_functions = [
            scaling.SimpleScalingFunction(torch.tensor(1.0, device = device)) for 
            gv,pv in zip(grads,ics)
            ]
    ics = torch.tensor([sf.unscale(i) for i, sf in zip(ics, scale_functions)])

    print("Initial parameter values:")
    for n, ic, sf in zip(names, ics, scale_functions):
        print("%s:\t%3.2e -> %3.2f" % (n, ic, sf.scale(ic)))
    print("")

    # 5) Create the actual model
    model = optimize.DeterministicModel(lambda *args: make(*args, 
        scale_functions = scale_functions), names, ics)

    lossv = loss(model(data, cycles, types, control), results)
    lossv.backward()
    print("Scaled parameter gradients:")
    grads = []
    for n in names:
        grads.append(getattr(model,n).grad.detach())
        print("%s:\t%.3e" % (n, grads[-1]))
    print("")

    # 6) Setup the optimizer
    niter = 50
    lr = 5.0e-3
    max_norm = 1.0e2
    optim = torch.optim.Adam(model.parameters(), lr = lr)

    # 7) Actually do the optimization!
    def closure():
        optim.zero_grad()
        pred = model(data, cycles, types, control)
        lossv = loss(pred, results)
        lossv.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        return lossv

    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_history = []
    for i in t:
        closs = optim.step(closure)
        loss_history.append(closs.detach().cpu().numpy())
        t.set_description("Loss: %3.2e" % loss_history[-1])

    # 8) Check accuracy of the optimized parameters
    print("")
    print("Optimized parameter accuracy:")
    for n, sf, tp in zip(names, scale_functions, params_true):
        print("%s:\t%3.2f/%3.2f" % (n, sf.scale(getattr(model, n).data), tp))

    # 9) Plot the convergence history
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
