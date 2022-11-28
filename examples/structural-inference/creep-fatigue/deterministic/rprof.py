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

from torch.profiler import profile, record_function, ProfilerActivity

from maker import make_model, load_subset_data

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
device = torch.device(dev)

# Maker function returns the ODE model given the parameters
# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, C, g, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model(
        torch.tensor(0.5), n, eta, s0, R, d, C, g, device=device, **kwargs
    ).to(device)


if __name__ == "__main__":
    # 1) Load the data for the variance of interest,
    #    cut down to some number of samples, and flatten
    scale = 0.0
    nsamples = 2  # 20 is the full number of samples in the default dataset
    input_data = xr.open_dataset(os.path.join("..", "scale-%3.2f.nc" % scale))
    data, results, cycles, types, control = load_subset_data(
        input_data, nsamples, device=device
    )
    ntime = 10
    data = data[:,:ntime]
    results = results[:ntime]
    cycles = cycles[:ntime]

    # 2) Setup names for each parameter and the initial conditions
    names = ["n", "eta", "s0", "R", "d", "C", "g"]
    ics = [torch.tensor(ra.uniform(0.25, 0.75), device=device) for i in range(5)] + [
        torch.tensor(ra.uniform(0.25, 0.75, size=(3,)), device=device),
        torch.tensor(ra.uniform(0.25, 0.75, size=(3,)), device=device),
    ]

    print("Initial parameter values:")
    for n, ic in zip(names, ics):
        print(("%s:\t" % n) + str(ic))
    print("")

    # 3) Create the actual model
    model = optimize.DeterministicModel(make, names, ics)

    # 4) Setup the optimizer
    niter = 1
    lr = 1.0e-2
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # 5) Setup the objective function
    loss = torch.nn.MSELoss(reduction="sum")

    # 6) Actually do the optimization!
    def closure():
        optim.zero_grad()
        pred = model(data, cycles, types, control)
        lossv = loss(pred, results)
        lossv.backward()
        return lossv
    
    #with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA], with_stack = True, on_trace_ready=torch.profiler.tensorboard_trace_handler('tb_results')) as prof:
    #with torch.autograd.profiler.profile(with_stack=True) as prof:
    t = tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_history = []
    for i in t:
        closs = optim.step(closure)
        loss_history.append(closs.detach().cpu().numpy())
        t.set_description("Loss: %3.2e" % loss_history[-1])
    
    #prof.export_stacks("profiler_stacks_cpu.txt", "self_cpu_time_total")
    #prof.export_stacks("profiler_stacks_gpu.txt", "self_cuda_time_total")


    # 7) Check accuracy of the optimized parameters
    print("")
    print("Optimized parameter accuracy (target values are all 0.5):")
    for n in names:
        print(("%s:\t" % n) + str(getattr(model, n).data))

