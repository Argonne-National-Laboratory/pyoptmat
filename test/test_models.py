import unittest

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn

from pyoptmat import models, flowrules, utility, hardening, damage
from pyoptmat.temperature import ConstantParameter as CP

torch.set_default_dtype(torch.float64)

class CommonModel:
  def test_derivs(self):
    v, dv = self.model.forward(self.t, self.state)
    ddv = utility.new_differentiate(lambda x: self.model.forward(self.t, x)[0], 
        self.state)
    
    self.assertTrue(np.allclose(dv,ddv, rtol = 1e-4,atol=1e-4))

class TestPerfectViscoplasticity(unittest.TestCase, CommonModel):
  def setUp(self):
    self.E = torch.tensor(100000.0)
    self.n = torch.tensor(5.2)
    self.eta = torch.tensor(110.0)
    
    self.times = torch.transpose(
        torch.tensor([np.linspace(0,1,4) for i in range(3)]), 1, 0)
    self.strains = torch.transpose(
        torch.tensor([np.linspace(0,1,4) for i in range(3)]), 1, 0) / 10.0
    self.temperatures = torch.zeros_like(self.strains)

    self.state = torch.tensor([[90.0],[100.0],[101.0]])
    self.t = self.times[2]

    self.flowrule = flowrules.PerfectViscoplasticity(CP(self.n), CP(self.eta))
    self.model = models.InelasticModel(CP(self.E), self.flowrule, 
        method = 'backward-euler')

    self.model._setup(self.times, self.strains, self.temperatures)

class TestIsoKinViscoplasticity(unittest.TestCase, CommonModel):
  def setUp(self):
    self.E = torch.tensor(100000.0)
    self.n = torch.tensor(5.2)
    self.eta = torch.tensor(110.0)
    self.s0 = torch.tensor(0.0)

    self.R = torch.tensor(101.0)
    self.d = torch.tensor(1.3)
    self.iso = hardening.VoceIsotropicHardeningModel(CP(self.R), CP(self.d))

    self.C = torch.tensor(12000.0)
    self.g = torch.tensor(10.1)
    self.kin = hardening.FAKinematicHardeningModel(CP(self.C), CP(self.g))

    self.flowrule = flowrules.IsoKinViscoplasticity(CP(self.n), CP(self.eta), 
        CP(self.s0), self.iso, self.kin)
    self.model = models.InelasticModel(CP(self.E), self.flowrule,
        method = 'backward-euler')

    self.times = torch.transpose(
        torch.tensor([np.linspace(0,1,4) for i in range(3)]), 1, 0)
    self.strains = torch.transpose(
        torch.tensor([np.linspace(0,1,4) for i in range(3)]), 1, 0)
    self.temperatures = torch.zeros_like(self.times)

    self.state = torch.tensor([[90.0,30.0,10.0,0],[100.0,10.0,15.0,0],[101.0,50.0,60.0,0]])/3

    self.t = self.times[2]

    self.model._setup(self.times, self.strains, self.temperatures)

class TestDamage(unittest.TestCase, CommonModel):
  def setUp(self):
    self.E = torch.tensor(100000.0)
    self.n = torch.tensor(5.2)
    self.eta = torch.tensor(110.0)
    self.s0 = torch.tensor(0.0)

    self.R = torch.tensor(101.0)
    self.d = torch.tensor(1.3)
    self.iso = hardening.VoceIsotropicHardeningModel(CP(self.R), CP(self.d))

    self.C = torch.tensor(1200.0)
    self.g = torch.tensor(10.1)
    self.kin = hardening.FAKinematicHardeningModel(CP(self.C), CP(self.g))

    self.A = torch.tensor(3000.0)
    self.xi = torch.tensor(6.5)
    self.phi = torch.tensor(1.7)
    self.dmodel = damage.HayhurstLeckie(CP(self.A), CP(self.xi), CP(self.phi))

    self.flowrule = flowrules.IsoKinViscoplasticity(CP(self.n), CP(self.eta), 
        CP(self.s0), self.iso, self.kin)
    self.model = models.InelasticModel(CP(self.E), self.flowrule,
        method = 'backward-euler', dmodel = self.dmodel)

    self.times = torch.transpose(
        torch.tensor([np.linspace(0,1,4) for i in range(3)]), 1, 0)
    self.strains = torch.transpose(
        torch.tensor([np.linspace(0,1,4) for i in range(3)]), 1, 0)
    self.temperatures = torch.zeros_like(self.strains)

    self.state = torch.tensor([[90.0,30.0,10.0,0.05],[100.0,10.0,15.0,0.1],[20,-10.0,-10,0.2]])
    self.t = self.times[2]

    self.model._setup(self.times, self.strains, self.temperatures)

class TestAll(unittest.TestCase, CommonModel):
  def setUp(self):
    self.E = torch.tensor(100000.0)
    self.n = torch.tensor(5.2)
    self.eta = torch.tensor(110.0)
    self.s0 = torch.tensor(0.0)

    self.R = torch.tensor(101.0)
    self.d = torch.tensor(1.3)
    self.iso = hardening.VoceIsotropicHardeningModel(CP(self.R), CP(self.d))

    self.C = torch.tensor([1200.0,200.0,10.0])
    self.g = torch.tensor([10.1,100.0,50.0])
    self.kin = hardening.ChabocheHardeningModel(CP(self.C), CP(self.g))

    self.A = torch.tensor(3000.0)
    self.xi = torch.tensor(6.5)
    self.phi = torch.tensor(1.7)
    self.dmodel = damage.HayhurstLeckie(CP(self.A), CP(self.xi), CP(self.phi))

    self.flowrule = flowrules.IsoKinViscoplasticity(CP(self.n), CP(self.eta), 
        CP(self.s0), self.iso, self.kin)
    self.model = models.InelasticModel(CP(self.E), self.flowrule,
        method = 'backward-euler', dmodel = self.dmodel)

    self.times = torch.transpose(
        torch.tensor([np.linspace(0,1,4) for i in range(3)]), 1, 0)
    self.strains = torch.transpose(
        torch.tensor([np.linspace(0,1,4) for i in range(3)]), 1, 0)
    self.temperatures = torch.zeros_like(self.strains)

    self.state = torch.tensor([[90.0,30.0,10.0,10.0,-10.0,0.2],[100.0,10.0,15.0,5.0,-10.0,0.3],[101.0,50.0,60.0,-50.0,10.0,0.4]])
    self.t = self.times[2]

    self.model._setup(self.times, self.strains, self.temperatures)
