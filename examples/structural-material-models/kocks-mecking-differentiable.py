#!/usr/bin/env python3

"""
    This example tests the Kocks-Mecking flow rule switching model
    by verifying we can switch from a viscoplastic to a rate independent
    response as a function of temperature and strain rate.
"""

import sys

sys.path.append("../..")

import torch
import numpy as np

import matplotlib.pyplot as plt

from pyoptmat import models, flowrules, experiments, hardening, temperature
from pyoptmat.temperature import ConstantParameter as CP

torch.set_default_tensor_type(torch.DoubleTensor)


def calculate_yield(strain, stress, offset=0.2 / 100.0):
    """
    Calculate the yield stress given a strain/stress history

    Args:
        strain (torch.tensor):      batched strains
        stress (torch.tensor):      batched stress
        offset (float, 0.002):      offset strain
    """
    E = (stress[1] - stress[0]) / (strain[1] - strain[0])

    vals = E * (strain - offset)

    first = (vals - stress) > 0

    vals, inds = torch.max(first, dim=0)

    return torch.diag(stress[inds])


if __name__ == "__main__":
    E = temperature.PolynomialScaling(
        [-6.82798735e-05, 9.48207244e-02, -1.11026526e02, 2.21183687e05]
    )
    mu = temperature.PolynomialScaling(
        [-2.60610204e-05, 3.61911162e-02, -4.23765368e01, 8.44212545e04]
    )
    g0 = torch.tensor(0.771)
    k = torch.tensor(1.38064e-20)
    b = torch.tensor(2.019e-7)
    eps0 = torch.tensor(1.0e6)

    A = torch.tensor(-3.35)
    B = torch.tensor(-3.23)
    C = torch.tensor(-5.82)

    n = temperature.KMRateSensitivityScaling(A, mu, b, k)
    eta = temperature.KMViscosityScaling(A, B, mu, eps0, b, k)
    s0 = temperature.ShearModulusScaling(torch.exp(C), mu)

    R = temperature.ConstantParameter(torch.tensor(100.0))
    d = temperature.ConstantParameter(torch.tensor(20.0))

    iso_hardening = hardening.VoceIsotropicHardeningModel(R, d)
    kin_hardening = hardening.NoKinematicHardeningModel()

    rd_flowrule = flowrules.IsoKinViscoplasticity(
        n,
        eta,
        temperature.ConstantParameter(torch.tensor(0.0)),
        iso_hardening,
        kin_hardening,
    )
    
    n_constant = CP(torch.tensor(15.0))
    eta_constant = CP(torch.tensor(1.0))
    ri_flowrule = flowrules.IsoKinViscoplasticity(
        n_constant, eta_constant, s0, iso_hardening, kin_hardening
    )
    
    sf = torch.tensor(10.0)
    flowrule = flowrules.SoftKocksMeckingRegimeFlowRule(
        ri_flowrule, rd_flowrule, g0, mu, b, eps0, k, sf
    )

    model = models.InelasticModel(E, flowrule)
    integrator = models.ModelIntegrator(model)

    ngrid = 10
    nsteps = 200
    elimits = torch.ones(ngrid) * 0.05

    # Constant temperature, varying flow rate
    erates = torch.logspace(-5, -9, ngrid)
    temps = torch.ones_like(erates) * (575 + 273.15)
    g_rate = k * temps / (mu.value(temps) * b**3.0) * torch.log(eps0 / erates)

    times, strains, temperatures, cyclces = experiments.make_tension_tests(
        erates, temps, elimits, nsteps
    )

    results_rate = integrator.solve_strain(times, strains, temperatures)
    yield_rate = calculate_yield(strains, results_rate[:, :, 0])
    norm_rate = yield_rate / mu.value(temps)

    # Constant flow rate, varying temperature
    temps = torch.linspace(400 + 273.15, 1000 + 273.15, ngrid)
    erates = torch.ones_like(temps) * 8.33e-5
    g_temp = k * temps / (mu.value(temps) * b**3.0) * torch.log(eps0 / erates)

    times, strains, temperatures, cyclces = experiments.make_tension_tests(
        erates, temps, elimits, nsteps
    )

    results_temp = integrator.solve_strain(times, strains, temperatures)
    yield_temp = calculate_yield(strains, results_temp[:, :, 0])
    norm_temp = yield_temp / mu.value(temps)

    plt.semilogy(
        g_rate.numpy(),
        norm_rate.numpy(),
        color="tab:blue",
        ls="none",
        marker="x",
        label="Varying rate",
    )
    plt.semilogy(
        g_temp.numpy(),
        norm_temp.numpy(),
        color="tab:orange",
        ls="none",
        marker="x",
        label="Varying temperature",
    )

    grange = np.linspace(0.4,1.4,50)
    plt.semilogy(
            grange,
            np.exp(C)*np.ones_like(grange),
            color = 'k',
            label = None)
    plt.semilogy(
            grange,
            np.exp(B.numpy()) * np.exp(A.numpy()*grange),
            color = 'k',
            label = None)

    plt.axvline(x=g0, ls="--", color="k")
    plt.legend(loc="best")
    plt.xlabel("Normalized activation energy")
    plt.ylabel("Normalized flow stress")
    plt.tight_layout()
    plt.show()
