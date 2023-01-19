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

from maker import make_model, downsample, nsize

from pyoptmat import optimize, experiments
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
def make(*args, **kwargs):
    """
        Maker with the Young's modulus fixed
    """
    return make_model(torch.tensor(0.5), *args, device=device, **kwargs).to(
        device
    )


if __name__ == "__main__":
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.01
    nsamples = 10  # at each strain rate
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = downsample(
        experiments.load_results(input_data, device=device),
        nsamples,
        input_data.nrates,
        input_data.nsamples,
    )

    # 2) Setup names for each parameter and the initial conditions
    names = ["n", "eta", "s0", "weights1", "biases1", "weights2", "biases2", "weights3", "biases3"]
    ics = [torch.tensor(ra.uniform(0, 1), device = device) for i in range(3)] + [
            torch.rand(nsize,1, device = device),
            torch.rand(nsize, device = device),
            torch.rand(nsize, nsize, device = device),
            torch.rand(nsize, device = device),
            torch.rand(1, nsize, device = device),
            torch.rand(1, device = device)]

    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print("%s" % n)
        print(ic)
    print("")

    # 3) Create the actual model
    model = optimize.DeterministicModel(make, names, ics)
    
    # 3.5) Make sure things aren't crazy
    test = model(data, cycles, types, control)
    plt.plot(data[-1].numpy(), results.numpy())
    plt.plot(data[-1].numpy(), test.detach().numpy(), ls = '--')
    plt.show()

    # 4) Setup the optimizer
    niter = 2000
    lr = 1.0e-3
    #optim = torch.optim.LBFGS(model.parameters(), history_size = 400, max_iter = 40)
    optim = torch.optim.AdamW(model.parameters(), lr)

    # 5) Setup the objective function
    loss = torch.nn.MSELoss(reduction="sum")

    # 6) Actually do the optimization!
    def closure():
        optim.zero_grad()
        pred = model(data, cycles, types, control)
        lossv = loss(pred, results)
        lossv.backward()
        return lossv

    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_history = []
    for i in t:
        closs = optim.step(closure)
        loss_history.append(closs.detach().cpu().numpy())
        t.set_description("Loss: %3.2e" % loss_history[-1])

    # 6.5) Make sure things aren't crazy
    test = model(data, cycles, types, control)
    plt.plot(data[-1].numpy(), results.numpy())
    plt.plot(data[-1].numpy(), test.detach().numpy(), ls = '--')
    plt.show()

    # 7) Check accuracy of the optimized parameters
    print("")
    print("Optimized parameter accuracy:")
    for n in names:
        print("%s" % n)
        print(getattr(model,n).data)

    # 8) Plot the convergence history
    plt.figure()
    plt.loglog(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.show()
