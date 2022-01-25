"""
  Various utility functions used in the rest of the modules.  This includes
  basic mathematical functions, routines used in the tests, and various
  visualization routines.
"""

import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as functional

from pyro.nn import PyroSample

def visualize_variance(strain, stress_true, stress_calc, alpha = 0.05):
  """
    Visualize variance for batched examples

    Args:
      strain (torch.tensor):        input strain
      stress_true (torch.tensor):   actual stress values
      stress_calc (torch.tensor):   simulated stress values

    Keyword Args:
      alpha (float): alpha value for shading
  """
  ntrue = stress_true.shape[1]
  max_true, _ = stress_true.kthvalue(int(ntrue*(1-alpha)), dim=1)
  min_true, _ = stress_true.kthvalue(int(ntrue*alpha), dim=1)
  mean_true = torch.mean(stress_true, 1)
  
  npred = stress_calc.shape[1]
  max_pred, _ = stress_calc.kthvalue(int(npred*(1-alpha)), dim=1)
  min_pred, _ = stress_calc.kthvalue(int(npred*alpha), dim=1)
  mean_pred = torch.mean(stress_calc, 1)

  plt.figure()
  plt.plot(strain.numpy(), mean_true.numpy(), 'k-', 
      label = "Actual mean")
  plt.fill_between(strain.numpy(), min_true.numpy(), max_true.numpy(), 
      alpha = 0.5, label = "Actual range")
  plt.plot(strain.numpy(), mean_pred.numpy(), 'k--', 
      label = "Predicted mean")
  plt.fill_between(strain.numpy(), min_pred.numpy(), max_pred.numpy(),
      alpha = 0.5, label = "Predicted range")
  plt.xlabel("Strain (mm/mm)")
  plt.ylabel("Stress (MPa)")
  plt.legend(loc='best')

  plt.show()

def new_differentiate(fn, x0, eps = 1.0e-6):
  """
    New numerical differentiation function to handle the batched-model cases

    Args:
      fn:               function to differentiate via finite differences
      x0:               point at which to take the numerical derivative
      eps (optional):   perturbation to use
  """
  v0 = fn(x0)
  nbatch = v0.shape[0]

  s1 = v0.shape[1:]
  if len(s1) == 0:
    s1 = (1,)
  fs1 = (nbatch,) + s1

  s2 = x0.shape[1:]
  if len(s2) == 0:
    flatten = True
    s2 = (1,)
  else:
    flatten = False
  fs2 = (nbatch,) + s2
  
  d = torch.empty((nbatch,) + s1 + s2)

  v0 = v0.reshape(fs1)

  for i2 in np.ndindex(s2):
    dx = torch.zeros_like(x0.reshape(fs2))
    inc = torch.abs(x0[...,i2]) * eps
    inc[inc < eps] = eps
    dx[...,i2] = inc

    if flatten:
      v1 = fn(x0 + dx[:,0]).reshape(fs1)
    else:
      v1 = fn(x0 + dx).reshape(fs1)
    
    d[:,:,i2] = torch.unsqueeze(((v1-v0)/inc),2)

  return d

def differentiate(fn, x0, eps = 1.0e-6):
  """
    Numerical differentiation used in the tests

    Args:
      fn:               function to differentiate via finite differences
      x0:               point at which to take the numerical derivative
      eps (optional):   perturbation to use
  """
  v0 = fn(x0)
  nbatch = v0.shape[0]
  s1 = v0.shape[1:]
  s2 = x0.shape[1:]
  
  if len(s1) == 0 and len(s2) == 0:
    d = torch.empty((nbatch,))
  elif len(s1) == 0:
    d = torch.empty((nbatch,) + s2)
  elif len(s2) == 0:
    d = torch.empty((nbatch,) + s1)
  else:
    d = torch.empty((nbatch,) + s1 + s2)

  for i2 in np.ndindex(s2):
    dx = torch.zeros_like(x0)
    if len(s2) == 0:
      dx = torch.abs(x0) * eps
    else:
      inc = torch.abs(x0[...,i2]) * eps
      inc[inc < eps] = eps
      dx[...,i2] = inc
    v1 = fn(x0 + dx)

    if len(s1) == 0 and len(s2) == 0:
      d = (v1-v0)/dx
    elif len(s2) == 0:
      d[...] = ((v1-v0)/dx.reshape(v1.shape))
    elif len(s1) == 0:
      d[...,i2] = ((v1-v0)/dx.reshape(v1.shape))[:,None]
    else:
      d[...,i2] = ((v1-v0)/dx)[...,None]

  return d

class BatchTimeSeriesInterpolator(nn.Module):
  """
    Precache a lot of the work required to interpolate in time vs
    the timeseries_interpolate_batch_times function

    Args:
      times:    input time series as a `(ntime,nbatch)` array
      values:   input values series as a `(ntime,nbatch)` array
  """
  def __init__(self, times, data):
    super().__init__()
    self.times = times
    self.values = data

    self.slopes = torch.diff(self.values, dim = 0) / torch.diff(self.times, 
        dim = 0)

  def forward(self, t):
    """
      Calculate the linearly-interpolated current values
    """
    gi = torch.remainder(torch.sum((self.times - t) <= 0, dim = 0), 
      self.times.shape[0])
    return torch.diagonal(self.values[gi-1]) + torch.diagonal(self.slopes[gi-1]) * (
        t - torch.diagonal(self.times[gi-1]))

class CheaterBatchTimeSeriesInterpolator(nn.Module):
  """
    Precache a lot of the work required to interpolate in time vs
    the timeseries_interpolate_batch_times function

    This is the cheater version specifically for our structured problems where
    the batches will always be at the same indices in time

    Args:
      times (torch.tensor):     input time series as a :code:`(ntime,nbatch)`
                                array
      values (torch.tensor):    input values series as a :code:`(ntime,nbatch)`
                                array
  """
  def __init__(self, times, data):
    super().__init__()
    self.times = times
    self.values = data

    self.slopes = torch.diff(self.values, dim = 0) / torch.diff(self.times, 
        dim = 0)

  def forward(self, t):
    """
      Calculate the linearly-interpolated current values
    """
    gi = torch.argmax((self.times[:,0] >= t[0]).type(torch.uint8))

    return self.values[gi-1] + self.slopes[gi-1] * (t - self.times[gi-1])

@torch.jit.script
def timeseries_interpolate_batch_times(times, values, t):
  """
    Interpolate the time series defined by X to the times defined by t

    Args:
      times     input time series as a `(ntime,nbatch)` array
      values    input value series as a `(ntime,nbatch)` array
      t         batch times as a `(nbatch,)` array

    Returns:
      Interpolated values as a `(nbatch,)` array
  """
  gi = torch.remainder(torch.sum((times - t) <= 0,dim = 0), times.shape[0])
  y2 = torch.diagonal(values[gi])
  y1 = torch.diagonal(values[gi-1])
  t2 = torch.diagonal(times[gi])
  t1 = torch.diagonal(times[gi-1])

  slopes = (y2 - y1) / (t2 - t1)
  return y1 + slopes * (t - t1)

def timeseries_interpolate_single_times(times, values, t):
  """
    Interpolate the time series defined by X to the times defined by t

    Args:
      times (torch.tensor):     input time series as a :math:`(ntime,)` array
      values (torch.tensor):    input value series as a :math:`(ntime,nbatch)`
                                array
      t (torch.tensor):         times as a scalar

    Returns:
      torch.tensor:             interpolated values as a :code:`(nbatch,)` array
  """
  gi = torch.remainder(torch.sum((times - t) <= 0,dim = 0), times.shape[0])
  slopes = (values[gi] - values[gi-1])/(times[gi,None] - times[gi-1,None])
  return values[gi-1] + slopes * (t - times[gi-1])

@torch.jit.script
def heaviside(X):
  """
    A pytorch-differentiable version of the Heaviside function

    .. math::
      
      H\\left(x\\right) = \\frac{\\operatorname{sign}(x) + 1)}{2}

    Args:
      X (torch.tensor): tensor input

    Returns:
      torch.tensor:     the Heaviside function of the input
  """
  return (torch.sign(X) + 1.0) / 2.0

@torch.jit.script
def macaulay(X):
  """
    A pytorch-differentiable version of the Macualay bracket

    .. math::
      
      M\\left(x\\right) = x H\\left(x\\right)

    Args:
      X (torch.tensor): tensor input

    Returns:
      torch.tensor:     the Macaulay bracket applied to the input
  """
  return X * heaviside(X)
