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

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from pyoptmat import models, flowrules, experiments, hardening
from pyoptmat.temperature import ConstantParameter as CP

torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    nrates = 5

    erates = torch.logspace(-1, -8, nrates)
    temps = torch.ones_like(erates)
    elimits = torch.ones_like(erates) * 0.1

    nsteps = 200

    times, strains, temperatures, cycles = experiments.make_tension_tests(
        erates, temps, elimits, nsteps
    )

    # Viscoplastic base model
    E = CP(torch.tensor(100000.0))

    n = CP(torch.tensor(15.0))
    eta = CP(torch.tensor(50.0))
    s0 = CP(torch.tensor(25.0))

    R = CP(torch.tensor(100.0))
    d = CP(torch.tensor(20.0))

    iso_hardening = hardening.VoceIsotropicHardeningModel(R, d)
    kin_hardening = hardening.NoKinematicHardeningModel()

    base_flowrule = flowrules.IsoKinViscoplasticity(
        n, eta, s0, iso_hardening, kin_hardening
    )

    flowrule = flowrules.IsoKinRateIndependentPlasticity(
        E, s0, iso_hardening, kin_hardening, s=10.0
    )

    base_model = models.InelasticModel(E, base_flowrule)
    base_integrator = models.ModelIntegrator(base_model, block_size=10, linesearch=True)

    rd_stresses = base_integrator.solve_strain(times, strains, temperatures)[:, :, 0]

    model = models.InelasticModel(E, flowrule)
    integrator = models.ModelIntegrator(model, block_size=10, linesearch=True)

    ri_stresses = integrator.solve_strain(times, strains, temperatures)[:, :, 0]

    def analytic(strain):
        Ev = E.value(0).numpy()
        s0v = s0.value(0).numpy()
        Rv = R.value(0).numpy()
        dv = d.value(0).numpy()

        pred = Ev * strain
        plastic = pred > s0v
        ystrain = s0v / Ev
        pstrain = strain[plastic] - ystrain
        pred[plastic] = s0v + Rv * (1.0 - np.exp(-dv * pstrain))

        return pred

    for i in range(nrates):
        l1 = plt.plot(strains[:, i], ri_stresses[:, i], "k-", lw=3)
        l2 = plt.plot(strains[:, i], rd_stresses[:, i], "k--", lw=3)

        enp = strains[:, i].numpy()
        exact = analytic(enp)
        l3 = plt.plot(enp, exact, color="r", lw=2, ls=":")

    plt.gca().legend([l1[0], l2[0], l3[0]], ["Approx RI", "RD", "Analytic"], loc="best")

    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")

    plt.show()
