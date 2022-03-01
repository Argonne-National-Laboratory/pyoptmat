#!/usr/bin/env python3

"""
  An example of how to load a model in stress control.  The example
  simulates the behavior of a simple viscoplastic model for the high
  temperature deformation of Alloy 617 developed by Messner, Phan,
  and Sham under creep conditions.  The simulations then load the
  model up to a given stress and then hold the model at that stress
  for a long period of time.  The experimental response is often
  given as a creep curve -- a plot of strain versus time during
  the hold at constant stress.  This example plots a modification, giving
  the total accumulated strain as a function of time, including the
  strain accumulated during the load up to the constant stress.
"""

import sys

sys.path.append("../..")

import torch

import matplotlib.pyplot as plt

from pyoptmat import models, flowrules, temperature, experiments, hardening

torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    # This example illustrates the temperature and rate sensitivity of
    # Alloy 617 with the model given in:
    # Messner, Phan, and Sham. IJPVP 2019.

    # Setup test conditions
    target_temperature = 950.0
    target_stresses = torch.linspace(80, 140, 13)
    target_times = torch.ones_like(target_stresses) * 100000 * 3600.0

    nbatch = len(target_stresses)
    nsteps_load = 50
    nsteps_hold = 100
    nsteps = nsteps_load + nsteps_hold

    loading_rate = 1.0

    times, stresses, temperatures, cycles = experiments.make_creep_tests(
        target_stresses,
        torch.ones_like(target_stresses) * target_temperature,
        torch.ones_like(target_stresses) * loading_rate,
        target_times,
        nsteps_load,
        nsteps_hold,
    )

    # Model with hardening
    A = torch.tensor(-8.679)
    B = torch.tensor(-0.744)
    mu = temperature.PolynomialScaling(
        torch.tensor([-1.34689305e-02, -5.18806776e00, 7.86708330e04])
    )
    b = torch.tensor(2.474e-7)
    k = torch.tensor(1.38064e-20)
    eps0 = torch.tensor(1e10)
    E = temperature.PolynomialScaling(
        torch.tensor([-3.48056033e-02, -1.44398964e01, 2.06464967e05])
    )

    ih = hardening.VoceIsotropicHardeningModel(
        temperature.ConstantParameter(torch.tensor(150.0)),
        temperature.ConstantParameter(torch.tensor(10.0)),
    )
    kh = hardening.NoKinematicHardeningModel()

    fr = flowrules.IsoKinViscoplasticity(
        temperature.KMRateSensitivityScaling(A, mu, b, k),
        temperature.KMViscosityScaling(A, B, mu, eps0, b, k),
        temperature.ConstantParameter(0),
        ih,
        kh,
    )
    model = models.InelasticModel(E, fr)
    integrator = models.ModelIntegrator(model)

    strains = integrator.solve_stress(times, stresses, temperatures)[:, :, 0]

    plt.plot(times / 3600.0, strains)
    plt.xlabel("Time (hrs)")
    plt.ylabel("Total strain (mm/mm)")
    plt.show()
