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
import numpy as np

import xarray as xr
import torch

from maker import make_model, downsample

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
def make(n, eta, s0, R, d, **kwargs):
    """
    Maker with the Young's modulus fixed
    """
    return make_model(torch.tensor(0.5), n, eta, s0, R, d, device=device, **kwargs).to(
        device
    )


if __name__ == "__main__":

    scales = [0.01, 0.15]
    samples = [5, 10, 20, 30]
    lrs = [0.01, 0.001, 0.1]

    for scale in scales:
        for sample in samples:
            for lr in lrs:
                # Write out the hyperparameter
                print("scale is: ", scale)
                print("sample is: ", sample)
                print("learning rate is: ", lr)
                # 1) Load the data for the variance of interest,
                #    cut down to some number of samples, and flatten
                scale = scale
                nsamples = sample  # at each strain rate
                input_data = xr.open_dataset(
                    os.path.join("..", "scale-%3.2f.nc" % scale)
                )
                data, results, cycles, types, control = downsample(
                    experiments.load_results(input_data, device=device),
                    nsamples,
                    input_data.nrates,
                    input_data.nsamples,
                )

                # 2) Setup names for each parameter and the initial conditions
                names = ["n", "eta", "s0", "R", "d"]
                ics = torch.tensor(
                    [ra.uniform(0, 1) for i in range(len(names))], device=device
                )

                print("Initial parameter values:")
                for n, ic in zip(names, ics):
                    print("%s:\t%3.2f" % (n, ic))
                print("")

                # 3) Create the actual model
                model = optimize.DeterministicModel(make, names, ics)

                # 4) Setup the optimizer
                niter = 200
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

                t = tqdm(range(niter), total=niter, desc="Loss:    ")
                loss_history = []
                for i in t:
                    closs = optim.step(closure)
                    loss_history.append(closs.detach().cpu().numpy())
                    t.set_description("Loss: %3.2e" % loss_history[-1])

                # 7) Check accuracy of the optimized parameters
                print("")
                print("Optimized parameter accuracy:")
                for n in names:
                    print("%s:\t%3.2f/0.50" % (n, getattr(model, n).data))

                # 8) Plot the convergence history
                np.savetxt(
                    "loss_adam_{}_{}_{}.txt".format(sample, scale, lr), loss_history
                )
                # plt.figure()
                # plt.plot(loss_history)
                # plt.xlabel("Iteration")
                # plt.ylabel("Loss")
                # plt.tight_layout()
                # plt.show()
