#!/usr/bin/env python3

import sys
sys.path.append('../..')

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
  temps = torch.linspace(750, 950, ntemps)+273.15
  elimits = torch.ones(ntemps) * 0.25
  erates = torch.ones(ntemps) * 8.33e-5

  nsteps = 200

  times, strains, temperatures = experiments.make_tension_tests(erates, temps, 
      elimits, nsteps)

  # Perfect viscoplastic model
  A = torch.tensor(-8.679)
  B = torch.tensor(-0.744)
  mu = temperature.PolynomialScaling(
      torch.tensor([-1.34689305e-02,-5.18806776e+00,7.86708330e+04]))
  b = torch.tensor(2.474e-7)
  k = torch.tensor(1.38064e-20)
  eps0 = torch.tensor(1e10)
  E = temperature.PolynomialScaling(
      torch.tensor([-3.48056033e-02, -1.44398964e+01, 2.06464967e+05]))

  
  ih = hardening.VoceIsotropicHardeningModel(
      temperature.ConstantParameter(torch.tensor(150.0)),
      temperature.ConstantParameter(torch.tensor(10.0)))
  kh = hardening.NoKinematicHardeningModel()

  fr = flowrules.IsoKinViscoplasticity(
      temperature.KMRateSensitivityScaling(A, mu, b, k),
      temperature.KMViscosityScaling(A, B, mu, eps0, b, k),
      temperature.ConstantParameter(0), 
      ih, kh)
  model = models.InelasticModel(E, fr)
  integrator = models.ModelIntegrator(model)

  stresses = integrator.solve_strain(times, strains, temperatures)[:,:,0]

  # Now do it backwards!
  strains_prime = integrator.solve_stress(times, stresses, temperatures)[:,:,0]

  for ei, epi, Ti, si in zip(strains.T.numpy(), strains_prime.T.numpy(),
      temperatures.T.numpy(), stresses.T.numpy()):
    l, = plt.plot(ei, si, label = "T = %3.0fK" % Ti[0])
    plt.plot(epi, si, label = None, ls = '--', color = l.get_color())
  
  plt.legend(loc='best')
  plt.xlabel("Strain (mm/mm)")
  plt.ylabel("Stress (Mpa)")
  plt.show()

