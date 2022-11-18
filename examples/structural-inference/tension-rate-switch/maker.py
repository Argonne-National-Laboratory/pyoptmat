#!/usr/bin/env python

"""
    Helper functions for the structural material model inference with
    tension tests examples.
"""

import sys

sys.path.append("../../..")

import numpy as np
import numpy.random as ra

import xarray as xr

import torch
from pyoptmat import models, flowrules, hardening, optimize, temperature
from pyoptmat.temperature import ConstantParameter as CP

from tqdm import tqdm

import tqdm

import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Actual parameters
A_true = -3.4
C_true = -5.9
g0_true = 0.55
R_true = 200.0
d_true = 5.0

# Various fixed parameters for the KM model
eps0 = 1e6
k = 1.380649e-20
b = 2.02e-7
eps0_ri = 1e-10
lmbda = 0.99

# Scale factor used in the model definition
sf = 0.5
sf_g = 0.1

# Constant temperature to run simulations at 
T = 550.0 + 273.15

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
dev = "cpu"
device = torch.device(dev)


def make_model(g0, A, C, R, d, device=torch.device("cpu"), **kwargs):
    """
    Key function for the entire problem: given parameters generate the model
    """
    E = temperature.PolynomialScaling(
        [-6.82798735e-05, 9.48207244e-02, -1.11026526e02, 2.21183687e05]
    )
    mu = temperature.PolynomialScaling(
        [-2.60610204e-05, 3.61911162e-02, -4.23765368e01, 8.44212545e04]
    )

    A_bound = optimize.bounded_scale_function(
            (
                torch.tensor(A_true * (1 - sf), device=device),
                torch.tensor(A_true * (1 + sf), device=device)
            )
            )
    C_bound = optimize.bounded_scale_function(
            (torch.tensor(C_true * (1 - sf), device=device),
            torch.tensor(C_true * (1 + sf), device=device))
            )
    g0_bound = optimize.bounded_scale_function(
            (torch.tensor(g0_true * (1-sf_g), device = device),
            torch.tensor(g0_true * (1+sf_g), device = device)))

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

    kinematic = hardening.NoKinematicHardeningModel()
    
    n = temperature.KMRateSensitivityScaling(A, mu, 
            torch.tensor(b, device = device), torch.tensor(k, device = device),
            A_scale = A_bound)
    eta = temperature.KMViscosityScalingGC(A,
            C, g0,
            mu, torch.tensor(eps0, device = device), 
            torch.tensor(b, device = device),
            torch.tensor(k, device = device),
            A_scale = A_bound, C_scale = C_bound, g0_scale = g0_bound)
    s0 = temperature.ShearModulusScalingExp(C, mu, A_scale = C_bound)

    rd_flowrule = flowrules.IsoKinViscoplasticity(n, eta,
        CP(torch.tensor(0.0, device = device)),
        isotropic,
        kinematic,
    )
    
    ri_flowrule_base = flowrules.IsoKinViscoplasticity(
            n, eta, s0, isotropic, kinematic)
    ri_flowrule = flowrules.RateIndependentFlowRuleWrapper(
            ri_flowrule_base, lmbda, eps0_ri)
    flowrule = flowrules.KocksMeckingRegimeFlowRule(
            ri_flowrule, rd_flowrule, 
            torch.tensor(g0, device = device),
            mu, 
            torch.tensor(b, device = device),
            torch.tensor(eps0, device = device),
            torch.tensor(k, device = device),
            g0_scale = g0_bound)

    model = models.InelasticModel(E, flowrule)

    return models.ModelIntegrator(model, **kwargs)


def generate_input(erates, emax, ntime):
    """
    Generate the times and strains given the strain rates, maximum strain, and number of time steps
    """
    strain = torch.repeat_interleave(
        torch.linspace(0, emax, ntime, device=device)[None, :], len(erates), 0
    ).T.to(device)
    time = strain / erates

    return time, strain


def downsample(rawdata, nkeep, nrates, nsamples):
    """
    Return fewer than the whole number of samples for each strain rate
    """
    ntime = rawdata[0].shape[1]
    return tuple(
        data.reshape(data.shape[:-1] + (nrates, nsamples))[..., :nkeep].reshape(
            data.shape[:-1] + (-1,)
        )
        for data in rawdata
    )


if __name__ == "__main__":
    # Running this script will regenerate the data
    ntime = 200
    emax = 0.5
    erates = torch.logspace(-2, -8, 7, device=device)
    nrates = len(erates)
    nsamples = 50

    scales = [0.0, 0.01, 0.05, 0.1, 0.15]

    times, strains = generate_input(erates, emax, ntime)

    for scale in scales:
        print("Generating data for scale = %3.2f" % scale)
        full_times = torch.empty((ntime, nrates, nsamples), device=device)
        full_strains = torch.empty_like(full_times)
        full_stresses = torch.empty_like(full_times)
        full_temperatures = torch.full_like(full_strains, T)

        for i in tqdm.tqdm(range(nsamples)):
            full_times[:, :, i] = times
            full_strains[:, :, i] = strains

            # True values are 0.5 with our scaling so this is easy
            model = make_model(
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
            )

            with torch.no_grad():
                full_stresses[:, :, i] = model.solve_strain(
                    times, strains, full_temperatures[:, :, i]
                )[:, :, 0]

        full_cycles = torch.zeros_like(full_times, dtype=int, device=device)
        types = np.array(["tensile"] * (nsamples * len(erates)))
        controls = np.array(["strain"] * (nsamples * len(erates)))

        ds = xr.Dataset(
            {
                "time": (["ntime", "nexp"], full_times.flatten(-2, -1).cpu().numpy()),
                "strain": (
                    ["ntime", "nexp"],
                    full_strains.flatten(-2, -1).cpu().numpy(),
                ),
                "stress": (
                    ["ntime", "nexp"],
                    full_stresses.flatten(-2, -1).cpu().numpy(),
                ),
                "temperature": (
                    ["ntime", "nexp"],
                    full_temperatures.cpu().flatten(-2, -1).numpy(),
                ),
                "cycle": (["ntime", "nexp"], full_cycles.flatten(-2, -1).cpu().numpy()),
                "type": (["nexp"], types),
                "control": (["nexp"], controls),
            },
            attrs={"scale": scale, "nrates": nrates, "nsamples": nsamples},
        )

        ds.to_netcdf("scale-%3.2f.nc" % scale)
