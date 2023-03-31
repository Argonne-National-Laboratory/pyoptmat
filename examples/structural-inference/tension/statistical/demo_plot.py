#!/usr/bin/env python3

import sys
import os.path

sys.path.append("../../../..")
sys.path.append("..")

import xarray as xr
import torch

from maker import make_model, downsample

from pyoptmat import optimize, experiments
from tqdm import trange

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Run on this on the cpu
dev = "cpu"
device = torch.device(dev)

# Don't try to optimize for the Young's modulus
def make(n, eta, s0, R, d, **kwargs):
    """
        Maker with the Young's modulus fixed
    """
    return make_model(torch.tensor(0.5), n, eta, s0, R, d, device=device, **kwargs).to(
        device
    )


if __name__ == "__main__":
    # Number of vectorized time steps
    time_chunk_size = 40

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

    names = ["n", "eta", "s0", "R", "d"]
    sampler = optimize.StatisticalModel(
        lambda *args, **kwargs: make(*args, block_size = time_chunk_size, **kwargs),
        names,
        [0.50, 0.49, 0.49, 0.48, 0.48],
        [0.02, 0.02, 0.03, 0.05, 0.05],
        torch.tensor(1.0e-4),
    )

    plt.figure()
    plt.plot(data[2, :, :nsamples].cpu(), results[:, :nsamples].cpu(), "k--")

    nsamples = 100
    alpha = 0.05 / 2

    times, strains, temps, cycles = experiments.make_tension_tests(
        torch.tensor([1.0e-2]), torch.tensor([0]), torch.tensor([0.5]), 200
    )
    data = torch.stack((times, temps, strains))
    control = torch.zeros(1, dtype=int)
    types = torch.zeros(1, dtype=int)

    stress_results = torch.zeros(nsamples, data.shape[1])

    for i in trange(nsamples):
        stress_results[i, :] = sampler(data, cycles, types, control)[:, 0]

    mean_result = torch.mean(stress_results, 0)
    sresults, _ = torch.sort(stress_results, 0)
    min_result = sresults[int(alpha * nsamples), :]
    max_result = sresults[int((1 - alpha) * nsamples), :]

    (l,) = plt.plot(data[2, :, 0], mean_result, lw=4, color="k")
    p = plt.fill_between(data[2, :, 0], min_result, max_result, alpha=0.5, color="k")

    plt.legend(
        [
            Line2D([0], [0], color="k", ls="--"),
            Line2D([0], [0], color="k", lw=4),
            Patch(facecolor="k", edgecolor=None, alpha=0.5),
        ],
        ["Experimental data", "Model average", "Model 95% prediction interval"],
        loc="best",
    )

    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.show()
