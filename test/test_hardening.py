import unittest

import numpy as np

import torch

from pyoptmat import hardening, utility

torch.set_default_dtype(torch.float64)

class HardeningBase:
  def test_dvalue(self):
    exact = self.model.dvalue(self.h)
    numer = utility.new_differentiate(lambda x: self.model.value(x), self.h)

    self.assertTrue(np.allclose(exact,numer,rtol=1.0e-4))

  def test_dstress(self):
    exact = self.model.dhistory_rate_dstress(self.s, self.h, self.t, self.ep)
    numer = utility.new_differentiate(lambda x: 
        self.model.history_rate(x, self.h, self.t, self.ep), self.s)

    self.assertTrue(np.allclose(exact,numer[:,:,0],rtol=1.0e-4))

  def test_dhistory(self):
    exact = self.model.dhistory_rate_dhistory(self.s, self.h, self.t, self.ep)
    numer = utility.new_differentiate(lambda x: 
        self.model.history_rate(self.s, x, self.t, self.ep), self.h)

    self.assertTrue(np.allclose(exact,numer,rtol=1.0e-3))

  def test_derate(self):
    exact = self.model.dhistory_rate_derate(self.s, self.h, self.t, self.ep)
    numer = utility.new_differentiate(lambda x:
        self.model.history_rate(self.s, self.h, self.t, x), self.ep)

    self.assertTrue(np.allclose(exact,numer,rtol=1.0e-4))

class TestVoceIsotropicHardening(unittest.TestCase, HardeningBase):
  def setUp(self):
    self.R = torch.tensor(100.0)
    self.d = torch.tensor(1.2)
    self.model = hardening.VoceIsotropicHardeningModel(self.R, self.d)

    self.nbatch = 10

    self.s = torch.linspace(90,100,self.nbatch)
    self.h = torch.reshape(torch.linspace(50,110,self.nbatch), 
        (self.nbatch,1))
    self.t = torch.ones(self.nbatch)
    self.ep = torch.linspace(0.1,0.2,self.nbatch)

class TestFAKinematicHardening(unittest.TestCase, HardeningBase):
  def setUp(self):
    self.C = torch.tensor(100.0)
    self.g = torch.tensor(1.2)
    self.model = hardening.FAKinematicHardeningModel(self.C, self.g)

    self.nbatch = 10

    self.s = torch.linspace(90,100,self.nbatch)
    self.h = torch.reshape(torch.linspace(50,110,self.nbatch), 
        (self.nbatch,1))
    self.t = torch.ones(self.nbatch)
    self.ep = torch.linspace(0.1,0.2,self.nbatch)

class TestChabocheKinematicHardening(unittest.TestCase, HardeningBase):
  def setUp(self):
    self.C = torch.tensor([100.0,1000,1500])
    self.g = torch.tensor([1.2,100,50])
    self.model = hardening.ChabocheHardeningModel(self.C, self.g)

    self.nbatch = 10

    self.s = torch.linspace(90,100,self.nbatch)
    self.h = torch.reshape(torch.linspace(50,110,self.nbatch*len(self.C)), 
        (self.nbatch,len(self.C)))
    self.t = torch.ones(self.nbatch)
    self.ep = torch.linspace(0.1,0.2,self.nbatch)
