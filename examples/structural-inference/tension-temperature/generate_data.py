#!/usr/bin/env python3

"""
    Model generators and code to generate synthetic data
"""

import sys
sys.path.append('../../..')

import numpy as np
import torch
import xarray as xr

import tqdm

from pyoptmat import models, flowrules, hardening, temperature, experiments, optimize

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# True model parameters
# Elastic constants
E_poly = torch.tensor([-5.76107011e-05,  7.52478382e-02, -9.98116448e+01,  2.19193150e+05], device = device)
mu_poly = torch.tensor([-2.19888172e-05,  2.87205489e-02, -3.80960476e+01,  8.36615078e+04], device = device)

# Rate sensitivity
A = -3.35
B = -3.23

# Hardening (initial slope)
H = 0.1

# Hardening (saturation)
tau0 = 1000.0
Q = 800.0

# Various physical/reference constants
eps0 = torch.tensor(1.0e6, device = device)
b = torch.tensor(2.019e-7, device = device)
k = torch.tensor(1.38064e-20, device = device)

# Setup constants
E = temperature.PolynomialScaling(E_poly)
mu = temperature.PolynomialScaling(mu_poly)

def true_model(A, B, H, tau0, Q, device = torch.device("cpu"), **kwargs):
    """True analytic model

    Args:
        A (float): Kocks-Mecking slope
        B (float): Kocks-Mecking interscept
        H (float): fraction of shear modulus for initial hardening slope
        tau0 (float): athermal hardening strength
        Q (float): hardening activation energy
    """
    n = temperature.KMRateSensitivityScaling(A, mu, b, k)
    eta = temperature.KMViscosityScaling(A, B, mu, eps0, b, k)
    theta = temperature.ShearModulusScaling(H, mu)
    tau = temperature.InverseArrheniusScaling(tau0, Q)

    isotropic = hardening.Theta0VoceIsotropicHardeningModel(
            tau,
            theta
            )

    kinematic = hardening.NoKinematicHardeningModel()
    flowrule = flowrules.IsoKinViscoplasticity(
            n, 
            eta, 
            temperature.ConstantParameter(torch.tensor(0.0, device = device)),
            isotropic,
            kinematic
    )
    model = models.InelasticModel(
            E,
            flowrule
    )

    return models.ModelIntegrator(model, **kwargs).to(device)

if __name__ == "__main__":
    # Amount of variability
    sf = 0.075
    noise = 2.0
    elimit = 0.1

    # Number of repeat samples
    nsamples = 4

    # Integration chunk size
    time_chunk_size = 25

    # Inputs defining the loading conditions
    nrate = 5
    ntemp = 50
    strain_rates = np.logspace(-5, -2, nrate)
    temperatures = np.linspace(500+273.15,800+273.15, ntemp)

    # Number of steps
    nsteps = 75

    # Generate the list of rates and temperatures
    SR, T = np.meshgrid(strain_rates, temperatures)
    SR = SR.flatten()
    T = T.flatten()
    elimits = np.ones_like(T) * elimit
    
    # Make the actual input data
    times, strains, temps, cycles = experiments.make_tension_tests(SR, T, elimits, nsteps)
    data = torch.stack((times, temps,strains))
    control = torch.zeros((len(SR),), dtype = int)
    types = torch.zeros_like(control)

    # Setup the parameters
    locs = torch.tensor([A, B, H, tau0, Q], device = device)
    scales = sf * torch.abs(locs)
    noise = torch.tensor(noise, device = device)

    # Setup the statistical model
    names = ["A", "B", "H", "tau0", "Q"]
    model = optimize.StatisticalModel(
            lambda *args, **kwargs: true_model(*args, block_size = time_chunk_size, **kwargs),
            names,
            locs.to(device), scales.to(device), noise.to(device))
    
    # Data storage...
    full_stress = torch.zeros_like(strains.unsqueeze(0).repeat_interleave(nsamples, dim = 0))
    
    # Run forward samples
    for i in tqdm.trange(nsamples):
        with torch.no_grad():
            full_stress[i,...] = model(data.to(device), cycles.to(device), types.to(device), control.to(device))
    
    # Expand the input data
    full_times = times.unsqueeze(0).repeat_interleave(nsamples, dim = 0)
    full_strains = strains.unsqueeze(0).repeat_interleave(nsamples, dim = 0)
    full_temps = temps.unsqueeze(0).repeat_interleave(nsamples, dim = 0)
    full_cycles = cycles.unsqueeze(0).repeat_interleave(nsamples, dim = 0)
    full_control = control.unsqueeze(0).repeat_interleave(nsamples, dim = 0)
    full_types = types.unsqueeze(0).repeat_interleave(nsamples, dim = 0)

    string_control = np.array(["strain"] * full_control.flatten().shape[0])
    string_types = np.array(["tensile"] * full_types.flatten().shape[0])

    ds = xr.Dataset(
        {
            "time": (["ntime", "nexp"], full_times.transpose(0,1).flatten(-2,-1).cpu().numpy()),
            "strain": (
                ["ntime", "nexp"],
                full_strains.transpose(0,1).flatten(-2,-1).cpu().numpy(),
            ),
            "stress": (
                ["ntime", "nexp"],
                full_stress.transpose(0,1).flatten(-2,-1).cpu().numpy(),
            ),
            "temperature": (
                ["ntime", "nexp"],
                full_temps.transpose(0,1).flatten(-2,-1).cpu().numpy(),
            ),
            "cycle": (["ntime", "nexp"], full_cycles.transpose(0,1).flatten(-2,-1).cpu().numpy()),
            "type": (["nexp"], string_types),
            "control": (["nexp"], string_control),
        },
        attrs={"nrates": nrate, "nsamples": nsamples},
    )

    ds.to_netcdf("data.nc")
