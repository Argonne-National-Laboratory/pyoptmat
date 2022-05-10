#!/usr/bin/env python3

"""
    Helper functions for the structural material model inference with
    creep-fatigue test examples.

    Running this script will regenerate the random, synthetic data. 
    So do not run the script if you want get results consistent
    with the tutorial.
"""

import sys

sys.path.append("../../..")

import numpy as np
import numpy.random as ra

import xarray as xr

import torch
from pyoptmat import models, flowrules, hardening, experiments, optimize
from pyoptmat.temperature import ConstantParameter as CP
import itertools

from tqdm import tqdm

import tqdm

import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Actual parameters
E_true = 150000.0
R_true = 200.0
d_true = 5.0
n_true = 7.0
eta_true = 300.0
s0_true = 50.0
C_true = np.array([10000.0, 5000.0, 500.0])
g_true = np.array([200.0, 150.0, 5.0])

# Scale factor used in the model definition
sf = 0.5

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = "cpu"
device = torch.device(dev)


def make_model(E, n, eta, s0, R, d, C, g, device=torch.device("cpu"), **kwargs):
    """
    Key function for the entire problem: given parameters generate the model
    """
    isotropic = hardening.VoceIsotropicHardeningModel(
        CP(
            R,
            scaling=optimize.bounded_scale_function(
                (
                    torch.tensor(R_true * (1 - sf), device=device),
                    torch.tensor(R_true * (1 + sf), device=device),
                )
            ),
        ),
        CP(
            d,
            scaling=optimize.bounded_scale_function(
                (
                    torch.tensor(d_true * (1 - sf), device=device),
                    torch.tensor(d_true * (1 + sf), device=device),
                )
            ),
        ),
    )
    kinematic = hardening.ChabocheHardeningModel(
        CP(
            C,
            scaling=optimize.bounded_scale_function(
                (
                    torch.tensor(C_true * (1 - sf), device=device),
                    torch.tensor(C_true * (1 + sf), device=device),
                )
            ),
        ),
        CP(
            g,
            scaling=optimize.bounded_scale_function(
                (
                    torch.tensor(g_true * (1 - sf), device=device),
                    torch.tensor(g_true * (1 + sf), device=device),
                )
            ),
        ),
    )
    flowrule = flowrules.IsoKinViscoplasticity(
        CP(
            n,
            scaling=optimize.bounded_scale_function(
                (
                    torch.tensor(n_true * (1 - sf), device=device),
                    torch.tensor(n_true * (1 + sf), device=device),
                )
            ),
        ),
        CP(
            eta,
            scaling=optimize.bounded_scale_function(
                (
                    torch.tensor(eta_true * (1 - sf), device=device),
                    torch.tensor(eta_true * (1 + sf), device=device),
                )
            ),
        ),
        CP(
            s0,
            scaling=optimize.bounded_scale_function(
                (
                    torch.tensor(s0_true * (1 - sf), device=device),
                    torch.tensor(s0_true * (1 + sf), device=device),
                )
            ),
        ),
        isotropic,
        kinematic,
    )
    model = models.InelasticModel(
        CP(
            E,
            scaling=optimize.bounded_scale_function(
                (
                    torch.tensor(E_true * (1 - sf), device=device),
                    torch.tensor(E_true * (1 + sf), device=device),
                )
            ),
        ),
        flowrule,
    )

    return models.ModelIntegrator(model, **kwargs)


def load_subset_data(dataset, nsamples, device=device):
    """
    Load some subset of the total number of samples
    in the dataset and cast it in the form needed
    to train pyoptmat models.
    """
    times = torch.tensor(
        dataset["time"].values[:, :, :nsamples], device=device
    ).flatten(-2, -1)
    temps = torch.tensor(
        dataset["temperature"].values[:, :, :nsamples], device=device
    ).flatten(-2, -1)
    strains = torch.tensor(
        dataset["strain"].values[:, :, :nsamples], device=device
    ).flatten(-2, -1)
    stresses = torch.tensor(
        dataset["stress"].values[:, :, :nsamples], device=device
    ).flatten(-2, -1)

    data = torch.stack((times, temps, strains))

    cycles = torch.tensor(
        dataset["cycle"].values[:, :, :nsamples], device=device
    ).flatten(-2, -1)

    types = torch.tensor(
        [
            experiments.exp_map[t]
            for t in dataset["type"].values[:, :nsamples].flatten()
        ],
        device=device,
    )
    control = torch.tensor(
        [
            experiments.control_map[t]
            for t in dataset["control"].values[:, :nsamples].flatten()
        ],
        device=device,
    )

    return data, stresses, cycles, types, control


if __name__ == "__main__":
    # Running this script will regenerate the data

    # Maximum strain in the cycle
    max_strains = np.logspace(np.log10(0.002), np.log10(0.02), 5)

    # Tension hold in the cycle
    tension_holds = np.array([1e-6, 60.0, 5 * 60.0, 60 * 60.0])

    # The data will sample all combinations of max_strains and
    # tension_holds, with the following fixed parameters
    R = -1.0  # Fully reversed loading
    strain_rate = 1.0e-3  # Fixed strain rate
    compression_hold = 1.0e-6  # No compression hold
    temperature = 500.0  # Problem will be temperature independent

    # Discretization
    N = 5  # Number of cycles
    nsamples = 20  # Number of repeats of each test
    nload = 20  # Time steps during the load period of each test
    nhold = 20  # Time steps during the hold period of each test

    # Scale values to generate
    scales = [0.0, 0.01, 0.05, 0.1, 0.15]

    # Generate the input data for a *single* run through each test
    ntime = N * 2 * (2 * nload + nhold) + 1
    ntests = max_strains.shape[0] * tension_holds.shape[0]
    times = torch.zeros(ntime, ntests, device=device)
    strains = torch.zeros(ntime, ntests, device=device)
    cycles = torch.zeros(ntime, ntests, dtype=int, device=device)

    strain_range = []
    hold_times = []
    i = 0
    for max_e in max_strains:
        for t_hold in tension_holds:
            timesi, strainsi, cyclesi = experiments.sample_cycle_normalized_times(
                {
                    "max_strain": max_e / 2.0,
                    "R": R,
                    "strain_rate": strain_rate,
                    "tension_hold": t_hold,
                    "compression_hold": compression_hold,
                },
                N,
                nload,
                nhold,
            )
            times[:, i] = torch.tensor(timesi, device=device)
            strains[:, i] = torch.tensor(strainsi, device=device)
            cycles[:, i] = torch.tensor(cyclesi, device=device)
            i += 1
            strain_range.append(max_e)
            hold_times.append(t_hold)

    temperatures = torch.ones_like(times) * temperature

    for scale in scales:
        print("Generating data for scale = %3.2f" % scale)
        full_times = torch.empty((ntime, ntests, nsamples), device=device)
        full_strains = torch.empty_like(full_times)
        full_stresses = torch.empty_like(full_times)
        full_temperatures = torch.zeros_like(full_strains)
        full_cycles = torch.zeros_like(full_times, dtype=int)
        full_ranges = torch.empty(full_times.shape[1:])
        full_rates = torch.empty_like(full_ranges)
        full_holds = torch.empty_like(full_ranges)

        for i in tqdm.tqdm(range(nsamples)):
            full_times[:, :, i] = times
            full_strains[:, :, i] = strains
            full_temperatures[:, :, i] = temperatures
            full_cycles[:, :, i] = cycles
            full_ranges[:, i] = torch.tensor(strain_range, device=device)
            full_rates[:, i] = strain_rate
            full_holds[:, i] = torch.tensor(hold_times, device=device)

            # True values are 0.5 with our scaling so this is easy
            model = make_model(
                torch.tensor(0.5, device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale, size=(3,)), device=device),
                torch.tensor(ra.normal(0.5, scale, size=(3,)), device=device),
                device=device,
            )

            with torch.no_grad():
                full_stresses[:, :, i] = model.solve_strain(
                    times, strains, temperatures
                )[:, :, 0]

        full_cycles = torch.zeros_like(full_times, dtype=int, device=device)
        types = np.array(["direct_data"] * (nsamples * ntests)).reshape(
            ntests, nsamples
        )
        controls = np.array(["strain"] * (nsamples * ntests)).reshape(ntests, nsamples)

        # This example uses a nonstandard format that you can read with `experiments.load_results`
        # but makes it easer to downsample the data.
        # Use the load_subset_data function above to read in the data
        ds = xr.Dataset(
            {
                "time": (["ntime", "ntests", "nsamples"], full_times.cpu().numpy()),
                "strain": (
                    ["ntime", "ntests", "nsamples"],
                    full_strains.cpu().numpy(),
                ),
                "stress": (
                    ["ntime", "ntests", "nsamples"],
                    full_stresses.cpu().numpy(),
                ),
                "temperature": (
                    ["ntime", "ntests", "nsamples"],
                    full_temperatures.cpu().numpy(),
                ),
                "cycle": (["ntime", "ntests", "nsamples"], full_cycles.cpu().numpy()),
                "type": (["ntests", "nsamples"], types),
                "control": (["ntests", "nsamples"], controls),
                "strain_ranges": (["ntests", "nsamples"], full_ranges.numpy()),
                "strain_rates": (["ntests", "nsamples"], full_rates.numpy()),
                "hold_times": (["ntests", "nsamples"], full_holds.numpy()),
            },
            attrs={"scale": scale},
        )

        ds.to_netcdf("scale-%3.2f.nc" % scale)
