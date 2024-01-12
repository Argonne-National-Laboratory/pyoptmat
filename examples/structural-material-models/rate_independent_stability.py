#!/usr/bin/env python3

"""
    This problem verifies the rate independent flow rule wrapper by comparing
    a numerically integrated solution to the exact, analytic solution for a
    simple load history.
"""

import sys

sys.path.append("../..")

import torch
import numpy as np

import matplotlib.pyplot as plt

from pyoptmat import models, flowrules, experiments, hardening
from pyoptmat.temperature import ConstantParameter as CP

torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    # Simple stability test for rate independent model

    # Setup test conditions
    nrates = 5
    elimits = torch.ones(nrates) * 0.25
    erates = torch.logspace(-8, -1, nrates)
    temps = np.ones_like(erates)

    nsteps = 200

    times, strains, temperatures, cycles = experiments.make_tension_tests(
        erates, temps, elimits, nsteps
    )

    # Viscoplastic base model
    E = CP(torch.tensor(100000.0))

    s0 = CP(torch.tensor(100.0))
    R = CP(torch.tensor(100.0))
    d = CP(torch.tensor(20.0))

    iso_hardening = hardening.VoceIsotropicHardeningModel(R, d)
    kin_hardening = hardening.NoKinematicHardeningModel()
    fr = flowrules.IsoKinRateIndependentPlasticity(
        E, s0, iso_hardening, kin_hardening, s=0.5
    )
    model = models.InelasticModel(E, fr)
    integrator = models.ModelIntegrator(model, linesearch=True, block_size=20)

    stresses = integrator.solve_strain(times, strains, temperatures)[:, :, 0]

    # Now do it backwards!
    strains_prime = integrator.solve_stress(times, stresses, temperatures)[:, :, 0]

    plt.figure()
    for i, (ei, epi, Ti, si) in enumerate(
        zip(
            strains.T.numpy(),
            strains_prime.T.numpy(),
            temperatures.T.numpy(),
            stresses.T.numpy(),
        )
    ):
        (l,) = plt.plot(ei, si, label=r"$\dot{\varepsilon}$ = %3.0e 1/s" % erates[i])
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
    big_types = torch.zeros(2 * nrates)
    big_types[nrates:] = 1

    both_results = integrator.solve_both(
        big_times, big_temperatures, big_data, big_types
    )

    plt.figure()
    for i in range(nrates):
        (l,) = plt.plot(
            strains[:, i],
            both_results[:, i, 0],
            label=r"$\dot{\varepsilon}$ = %3.0e 1/s" % erates[i],
        )
        plt.plot(
            both_results[:, i + nrates, 0],
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

    nstress = 5
    input_stress = torch.linspace(50, 150, nstress)
    times, stresses, temperatures, cycles = experiments.make_creep_tests(
        input_stress,
        torch.ones_like(input_stress),
        torch.ones_like(input_stress),
        torch.ones_like(input_stress) * 1000.0,
        50,
        50,
    )

    creep_strains = integrator.solve_stress(times, stresses, temperatures)[:, :, 0]

    plt.figure()
    plt.plot(times, creep_strains, label=["%i MPa" % s for s in input_stress])
    plt.xlabel("Time (s)")
    plt.ylabel("Creep strain (mm/mm)")
    plt.legend(loc="best")
    plt.show()
