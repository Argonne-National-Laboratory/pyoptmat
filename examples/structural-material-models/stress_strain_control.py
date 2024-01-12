#!/usr/bin/env python3

"""
  This example demonstrates the ability of pyoptmat to integrate material models under
  strain control, stress control, or a combination of both.

  The example first simulates standard, strain-rate controlled tensile curves
  for a simple material model for Alloy 617 developed by Messner, Phan, and Sham.  These
  simulations then take time, strain, and temperature as inputs and outputs the
  resulting stresses.

  Then, the example uses the stresses from the strain controlled simulations
  to repeat the simulations under stress control.  These simulations therefore
  take as input time, temperature, and stress and output strain.  The plot demonstrates that
  the strain controlled and stress controlled simulations produce identical results.

  Finally, the example integrates the model twice for each temperature, once in strain
  control and once in stress control, but integrated all the experiments at once.
  This example then demos the ability of pyoptmat to integrate both stress and strain
  controlled tests in the same vectorized calculation.
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
    ntemps = 5
    temps = torch.linspace(750, 950, ntemps) + 273.15
    elimits = torch.ones(ntemps) * 0.25
    erates = torch.ones(ntemps) * 8.33e-5

    nsteps = 200

    times, strains, temperatures, cycles = experiments.make_tension_tests(
        erates, temps, elimits, nsteps
    )

    # Perfect viscoplastic model
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

    stresses = integrator.solve_strain(times, strains, temperatures)[:, :, 0]

    # Now do it backwards!
    strains_prime = integrator.solve_stress(times, stresses, temperatures)[:, :, 0]

    plt.figure()
    for ei, epi, Ti, si in zip(
        strains.T.numpy(),
        strains_prime.T.numpy(),
        temperatures.T.numpy(),
        stresses.T.numpy(),
    ):
        (l,) = plt.plot(ei, si, label="T = %3.0fK" % Ti[0])
        plt.plot(epi, si, label=None, ls="--", color=l.get_color(), lw=4, alpha=0.5)

    plt.legend(loc="best")
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (Mpa)")
    plt.title("Comparison between strain and stress control")
    plt.show()

    # Now do it both ways at once!
    big_times = torch.cat((times, times), 1)
    big_temperatures = torch.cat((temperatures, temperatures), 1)
    big_data = torch.cat((strains, stresses), 1)
    big_types = torch.zeros(2 * ntemps)
    big_types[ntemps:] = 1

    both_results = integrator.solve_both(
        big_times, big_temperatures, big_data, big_types
    )

    plt.figure()
    for i in range(ntemps):
        (l,) = plt.plot(
            strains[:, i], both_results[:, i, 0], label="T = %3.0fK" % Ti[i]
        )
        plt.plot(
            both_results[:, i + ntemps, 0],
            stresses[:, i],
            label=None,
            ls="--",
            color=l.get_color(),
            lw=4,
            alpha=0.5,
        )

    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.legend(loc="best")
    plt.title("Running both strain and stress control at once")
    plt.show()
