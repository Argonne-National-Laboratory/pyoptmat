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

torch.set_default_tensor_type(torch.DoubleTensor)

import matplotlib.pyplot as plt

from pyoptmat import models, flowrules, temperature, experiments, hardening, damage
from pyoptmat.temperature import ConstantParameter as CP

torch.set_default_tensor_type(torch.DoubleTensor)

if __name__ == "__main__":
    E = CP(torch.tensor(130000.0))
    n = CP(torch.tensor(6.0))
    eta = CP(torch.tensor(600.0))
    s0 = CP(torch.tensor(0.0))
    theta0 = CP(torch.tensor(3000.0))
    tau = CP(torch.tensor(100.0))
   
    isotropic = hardening.Theta0VoceIsotropicHardeningModel(tau, theta0)
    kinematic = hardening.NoKinematicHardeningModel()
    flowrule = flowrules.IsoKinViscoplasticity(n, eta, 
            s0, isotropic, kinematic)
    
    C = CP(torch.tensor(15.94))
    A = CP(torch.tensor(-5379.0))
    B = CP(torch.tensor(29316.3))

    dmodel = damage.LarsonMillerDamage(C, A, B)

    model = models.DamagedInelasticModel(E, flowrule, dmodel = dmodel)

    integrator = models.ModelIntegrator(model, block_size = 20)

    # Creep
    target_temperature = 550.0 + 273.15
    target_stresses = torch.tensor([180.0])
    target_times = torch.ones_like(target_stresses) * 15000

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

    state = integrator.solve_stress(times, stresses, temperatures)

    strains = state[...,0]

    plt.plot(times, strains)
    plt.xlabel("Time (hrs)")
    plt.ylabel("Total strain (mm/mm)")
    plt.show()

    plt.plot(times, stresses)
    plt.xlabel("Time (hrs)")
    plt.ylabel("Stress (MPa)")
    plt.show()

    plt.plot(times, state[...,-1])
    plt.xlabel("Time (hrs)")
    plt.ylabel("Damage")
    plt.show()
    
    # Simpler 
    R = CP(torch.tensor(1.0e-3))
    dmodel = damage.ConstantDamage(R)
    model = models.DamagedInelasticModel(E, flowrule, dmodel = dmodel)
    integrator = models.ModelIntegrator(model, block_size = 20)

    # Tension with unload
    times = torch.cat([torch.linspace(0, 750, nsteps_hold) , torch.linspace(750, 751, nsteps_hold)[1:]]).unsqueeze(-1)
    strains = torch.cat([torch.linspace(0, 0.4, nsteps_hold), torch.linspace(0.4, 0.398, nsteps_hold)[1:]]).unsqueeze(-1)
    temperatures = torch.zeros_like(strains)

    state = integrator.solve_strain(times, strains, temperatures)
    stress = state[...,0]

    E_final = (stress[-2,0] - stress[-1,0])/(strains[-2,0] - strains[-1,0])
    d = state[-1,0,-1]

    print("Initial modulus: %f" % E.pvalue.cpu())
    print("Final modulus: %f" % E_final.cpu())
    print("Calculated modulus: %f" % (E.pvalue * (1-d)).cpu())

    plt.plot(strains, stress)
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.show()

