#!/usr/bin/env python3

import sys
sys.path.append('../..')

import torch

import matplotlib.pyplot as plt

from pyoptmat import models, flowrules, temperature, experiments

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


  fr = flowrules.PerfectViscoplasticity(
      temperature.KMRateSensitivityScaling(A, mu, b, k),
      temperature.KMViscosityScaling(A, B, mu, eps0, b, k))
  model = models.InelasticModel(E, fr, progress = True)

  stresses = model.solve(times, strains, temperatures)[:,:,0]

  print("Temperature scaling")
  for ei, Ti, si in zip(strains.T.numpy(), temperatures.T.numpy(), 
      stresses.T.numpy()):
    plt.plot(ei, si, label = "T = %3.0fK" % Ti[0])
  
  plt.legend(loc='best')
  plt.xlabel("Strain (mm/mm)")
  plt.ylabel("Stress (Mpa)")
  plt.show()

