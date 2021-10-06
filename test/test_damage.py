import unittest

import numpy as np

import torch

from pyoptmat import damage, utility
from pyoptmat.temperature import ConstantParameter as CP

torch.set_default_dtype(torch.float64)

class DamageBase:
  def test_d_rate(self):
    exact = self.model.damage_rate(self.s, self.d, self.t, self.T)[1]
    numer = utility.differentiate(
        lambda x: self.model.damage_rate(self.s, x, self.t, self.T)[0], 
        self.d)
    self.assertTrue(np.allclose(exact, numer))

  def test_d_stress(self):
    exact = self.model.d_damage_rate_d_s(self.s, self.d, self.t, self.T)
    numer = utility.differentiate(
        lambda x: self.model.damage_rate(x, self.d, self.t, self.T)[0], 
        self.s)
    self.assertTrue(np.allclose(exact, numer))

class TestNoDamage(unittest.TestCase, DamageBase):
  def setUp(self):
    self.model = damage.NoDamage()

    self.nbatch = 10

    self.s = torch.linspace(90,100,self.nbatch)
    self.d = torch.linspace(0.1,0.5, self.nbatch)
    self.t = torch.ones(self.nbatch)
    self.T = torch.zeros_like(self.t)

  def test_damage_rate(self):
    self.assertTrue(np.allclose(self.model.damage_rate(self.s, self.d, self.t, self.T)[0],
      torch.zeros_like(self.s)))

class TestHLDamage(unittest.TestCase, DamageBase):
  def setUp(self):
    self.A = 3000.0
    self.xi = 6.5
    self.phi = 1.7
    self.model = damage.HayhurstLeckie(CP(self.A), CP(self.xi), CP(self.phi))

    self.nbatch = 10

    self.s = torch.linspace(90,100,self.nbatch)
    self.d = torch.linspace(0.1,0.5, self.nbatch)
    self.t = torch.ones(self.nbatch)
    self.T = torch.zeros_like(self.t)

  def test_damage_rate(self):
    self.assertTrue(np.allclose(self.model.damage_rate(self.s, self.d, self.t, self.T)[0],
      (self.s/self.A)**(self.xi) * (1 - self.d)**(self.xi-self.phi)))
