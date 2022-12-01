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
    nrates = 5

    erates = torch.logspace(-1, -6, nrates)
    temps = torch.ones_like(erates)
    elimits = torch.ones_like(erates) * 0.1

    nsteps = 200

    times, strains, temperatures, cycles = experiments.make_tension_tests(
        erates, temps, elimits, nsteps
    )

    # Viscoplastic base model
    E = CP(torch.tensor(100000.0))

    n = CP(torch.tensor(5.0))
    eta = CP(torch.tensor(50.0))
    s0 = CP(torch.tensor(25.0))

    R = CP(torch.tensor(100.0))
    d = CP(torch.tensor(20.0))

    iso_hardening = hardening.VoceIsotropicHardeningModel(R, d)
    kin_hardening = hardening.NoKinematicHardeningModel()

    base_flowrule = flowrules.IsoKinViscoplasticity(
        n, eta, s0, iso_hardening, kin_hardening
    )

    # Approximately rate independent flow rule
    n_ri = CP(torch.tensor(15))
    eta_ri = CP(torch.tensor(1.0))
    flowrule = flowrules.IsoKinViscoplasticity(n_ri, eta_ri, s0,
            iso_hardening, kin_hardening) 

    base_model = models.InelasticModel(E, base_flowrule)
    base_integrator = models.ModelIntegrator(base_model)

    model = models.InelasticModel(E, flowrule)
    integrator = models.ModelIntegrator(model)
    
    ri_stresses = integrator.solve_strain(times, strains, temperatures)[:, :, 0]
    ri_strains = integrator.solve_stress(times, ri_stresses, temperatures)[:, :, 0]

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
        plt.plot(strains[:, i], ri_stresses[:, i], "k-", lw=3)
        plt.plot(ri_strains[:, i], ri_stresses[:, i], "b-", lw = 2)

        enp = strains[:, i].numpy()
        exact = analytic(enp)
        plt.plot(enp, exact, color="r", lw=2, ls=":")

    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")

    plt.show()
