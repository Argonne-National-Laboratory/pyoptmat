import unittest

import numpy as np

import torch
import torch.nn
from torch.nn import Parameter

from pyoptmat import ode, models, flowrules, hardening
from pyoptmat.temperature import ConstantParameter as CP

torch.set_default_tensor_type(torch.DoubleTensor)

class FallingODE(torch.nn.Module):
  def __init__(self, m, w, k):
    super().__init__()
    self.m = m
    self.w = w
    self.k = k

  def forward(self, t, y):
    f = torch.empty(y.shape)
    f[...,0] = y.clone()[...,1]
    f[...,1] = (self.w - self.k * y.clone()[...,1]) / self.m

    df = torch.zeros(y.shape+y.shape[-1:])
    df[...,0,1] = 1
    df[...,1,1] = -self.k / self.m

    return f, df

class TestFallingSensitivities(unittest.TestCase):
  def setUp(self):
    self.m = torch.nn.Parameter(torch.tensor(0.25))
    self.w = torch.nn.Parameter(torch.tensor(8.0))
    self.k = torch.nn.Parameter(torch.tensor(2.0))

    self.nbatch = 10
    self.y0 = torch.zeros(self.nbatch, 2)
    
    self.nsteps = 100
    self.times = torch.tensor(np.array([np.linspace(0, 5.0, self.nsteps) for i in range(self.nbatch)]).T)

    self.model = FallingODE(self.m, self.w, self.k)

    self.reduction = torch.nn.MSELoss(reduction='sum')

  def compare(self, **params):
    res1 = ode.odeint(self.model, self.y0, self.times,  
        **params)
    loss1 = self.reduction(res1, torch.ones_like(res1))
    lr1 = loss1.detach().numpy()
    self.model.zero_grad()
    loss1.backward()

    grads1 = np.array([self.m.grad.numpy(), self.w.grad.numpy(), self.k.grad.numpy()])
    
    self.model.zero_grad()

    res2 = ode.odeint_adjoint(self.model, self.y0, self.times, **params)
    loss2 = self.reduction(res2, torch.ones_like(res2))
    lr2 = loss2.detach().numpy()
    loss2.backward()

    grads2 = np.array([self.m.grad.numpy(), self.w.grad.numpy(), self.k.grad.numpy()])
    
    self.assertAlmostEqual(lr1, lr2)
    
    print(grads1)
    print(grads2)
    self.assertTrue(np.allclose(grads1, grads2,rtol=1.0e-4))

  def test_explicit(self):
    self.compare(method = "forward-euler")

  def test_explicit_substep(self):
    self.compare(method = "forward-euler", substep = 3)

  def test_implicit(self):
    self.compare(method = "backward-euler")

  def test_implicit_substep(self):
    self.compare(method = "backward-euler", substep = 3)

class TestFullModelStrain(unittest.TestCase):
  def setUp(self):
    self.E = 100000.0
    self.n = 5.2
    self.eta = 110.0
    self.R = 100.0
    self.d = 5.1
    self.C = 1000.0
    self.g = 10.0
    self.s0 = 10.0

    self.ntime = 100
    self.nbatch = 10

    self.times = torch.transpose(
        torch.tensor([np.linspace(0,1,self.ntime) for i in range(self.nbatch)]), 1, 0)
    self.strains = torch.transpose(
        torch.tensor([np.linspace(0,0.01,self.ntime) for i in range(self.nbatch)]), 1, 0)
    self.temperatures = torch.zeros_like(self.strains)

    self.reduction = torch.nn.MSELoss(reduction='sum')

  def make_model(self, **params):
    E = CP(Parameter(torch.tensor(self.E).detach()))
    n = CP(Parameter(torch.tensor(self.n).detach()))
    eta = CP(Parameter(torch.tensor(self.eta).detach()))
    R = CP(Parameter(torch.tensor(self.R).detach()))
    d = CP(Parameter(torch.tensor(self.d).detach()))
    C = CP(Parameter(torch.tensor(self.C).detach()))
    g = CP(Parameter(torch.tensor(self.g).detach()))
    s0 = CP(Parameter(torch.tensor(self.s0).detach()))

    return models.ModelIntegrator(models.InelasticModel(E, 
        flowrules.IsoKinViscoplasticity(n, eta, s0, 
          hardening.VoceIsotropicHardeningModel(R, d),
          hardening.FAKinematicHardeningModel(C, g))), **params), [E.pvalue, n.pvalue, eta.pvalue, R.pvalue, d.pvalue, C.pvalue, g.pvalue, s0.pvalue]

  def compare(self, **params):
    model1, params1 = self.make_model(use_adjoint = False, **params)
    res1 = model1.solve_strain(self.times, self.strains, self.temperatures)
    loss1 = self.reduction(res1, torch.ones_like(res1))
    lr1 = loss1.detach().numpy()
    model1.zero_grad()
    loss1.backward()
    gr1 = np.array([p.grad.numpy() for p in params1])

    model2, params2 = self.make_model(use_adjoint = True, **params)
    res2 = model2.solve_strain(self.times.detach().clone(), self.strains, self.temperatures)
    loss2 = self.reduction(res2, torch.ones_like(res2))
    lr2 = loss2.detach().numpy()
    model2.zero_grad()
    loss2.backward()
    gr2 = np.array([p.grad.numpy() for p in params2])
    
    self.assertAlmostEqual(lr1, lr2)
    self.assertTrue(np.allclose(gr1, gr2, rtol = 1.0e-2))

  def test_explicit(self):
    self.compare(method = "forward-euler")

  def test_explicit_substep(self):
    self.compare(method = "forward-euler", substeps = 2)

  def test_implicit(self):
    self.compare(method = "backward-euler")
  
  def test_implicit_substep(self):
    self.compare(method = "backward-euler", substeps = 2)

class TestFullerModelStrain(unittest.TestCase):
  def setUp(self):
    self.E = 100000.0
    self.n = 5.2
    self.eta = 110.0
    self.R = 100.0
    self.d = 5.1
    self.C = [1000.0, 500.0]
    self.g = [10.0, 2.5]
    self.s0 = 10.0

    self.ntime = 100
    self.nbatch = 10

    self.times = torch.transpose(
        torch.tensor([np.linspace(0,1,self.ntime) for i in range(self.nbatch)]), 1, 0)
    self.strains = torch.transpose(
        torch.tensor([np.linspace(0,0.01,self.ntime) for i in range(self.nbatch)]), 1, 0)
    self.temperatures = torch.zeros_like(self.strains)

    self.reduction = torch.nn.MSELoss(reduction='sum')

  def make_model(self, **params):
    E = CP(Parameter(torch.tensor(self.E).detach()))
    n = CP(Parameter(torch.tensor(self.n).detach()))
    eta = CP(Parameter(torch.tensor(self.eta).detach()))
    R = CP(Parameter(torch.tensor(self.R).detach()))
    d = CP(Parameter(torch.tensor(self.d).detach()))
    C = CP(Parameter(torch.tensor(self.C).detach()))
    g = CP(Parameter(torch.tensor(self.g).detach()))
    s0 = CP(Parameter(torch.tensor(self.s0).detach()))

    return models.ModelIntegrator(models.InelasticModel(E, 
        flowrules.IsoKinViscoplasticity(n, eta, s0, 
          hardening.VoceIsotropicHardeningModel(R, d),
          hardening.ChabocheHardeningModel(C, g))),
        **params), [E.pvalue, n.pvalue, eta.pvalue, R.pvalue, d.pvalue, C.pvalue, g.pvalue, s0.pvalue]

  def compare(self, **params):
    model1, params1 = self.make_model(use_adjoint = False, **params)
    res1 = model1.solve_strain(self.times, self.strains, self.temperatures)
    loss1 = self.reduction(res1, torch.ones_like(res1))
    lr1 = loss1.detach().numpy()
    model1.zero_grad()
    loss1.backward()
    gr1 = [p.grad.numpy() for p in params1]

    model2, params2 = self.make_model(use_adjoint = True, **params)
    res2 = model2.solve_strain(self.times.detach().clone(), self.strains, self.temperatures)
    loss2 = self.reduction(res2, torch.ones_like(res2))
    lr2 = loss2.detach().numpy()
    model2.zero_grad()
    loss2.backward()
    gr2 = [p.grad.numpy() for p in params2]

    self.assertAlmostEqual(lr1, lr2)
    for p1, p2 in zip(gr1, gr2):
      self.assertTrue(np.allclose(p1, p2, rtol = 1.0e-2))

  def test_explicit(self):
    self.compare(method = "forward-euler")

  def test_explicit_substep(self):
    self.compare(method = "forward-euler", substeps = 2)

  def test_implicit(self):
    self.compare(method = "backward-euler")
  
  def test_implicit_substep(self):
    self.compare(method = "backward-euler", substeps = 2)
