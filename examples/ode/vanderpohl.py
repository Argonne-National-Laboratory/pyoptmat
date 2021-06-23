#!/usr/bin/env python3

import sys

sys.path.append('../..')

import numpy as np
import torch

import matplotlib.pyplot as plt

from pyoptmat import ode

torch.set_default_tensor_type(torch.DoubleTensor)

class VanderPolODE(torch.nn.Module):
  """
    From Manichev et al 2019
      
      x0 = [-1, 1]
      t = [0, 4.2*mu)
  """
  def __init__(self, mu):
    super().__init__()
    self.mu = torch.tensor(mu)

  def forward(self, t, y):
    f = torch.empty(y.shape)
    f[...,0] = y[...,1]
    f[...,1] = -y[...,0] + self.mu * (1.0 - y[...,0]**2.0) * y[...,1]

    df = torch.empty(y.shape+y.shape[1:])
    df[...,0,0] = 0
    df[...,0,1] = 1
    df[...,1,0] = -1 - 2.0 * self.mu * y[...,0] * y[...,1]
    df[...,1,1] = self.mu * (1.0 - y[...,0]**2.0)

    return f, df

if __name__ == "__main__":
  # Common
  y0 = torch.tensor([[-1.0,1]])
  period = lambda m: (3.0 - 2*np.log(2)) * m + 2.0*np.pi/mu**(1.0/3)
  
  # Test for non-stiff version
  mu = 1.0
  times = torch.linspace(0, period(mu) * 10, 1000).reshape(1000,1)

  model = VanderPolODE(mu)

  res_exp = ode.odeint(model, y0, times, method = "forward-euler", 
      substep = 10)
  res_imp = ode.odeint(model, y0, times, method = "backward-euler",
      substep = 10, solver_method = "lu")
  
  plt.figure()
  plt.plot(times, res_exp[:,0,0], label = 'Forward')
  plt.plot(times, res_imp[:,0,0], label = 'Backward')
  plt.legend(loc='best')
  plt.show()

  # Test for moderately stiff version
  mu = 14.0
  times = torch.linspace(0, period(mu) * 10, 1000).reshape(1000,1)

  model = VanderPolODE(mu)

  res_exp = ode.odeint(model, y0, times, method = "forward-euler",
      substep = 10)
  res_imp = ode.odeint(model, y0, times, method = "backward-euler",
      substep = 10, solver_method = "lu")
  
  plt.figure()
  plt.plot(times, res_exp[:,0,0], label = 'Forward')
  plt.plot(times, res_imp[:,0,0], label = 'Backward')
  plt.legend(loc='best')
  plt.show()

  # Test for highly stiff version (explicit just explodes)
  mu = 30.0
  times = torch.linspace(0, period(mu) * 10 * 10.0/16, 1000).reshape(1000,1)

  model = VanderPolODE(mu)

  res_imp = ode.odeint(model, y0, times, method = "backward-euler",
      substep = 20, solver_method = "lu")
  
  plt.figure()
  plt.plot(times, res_imp[:,0,0])
  plt.show()
