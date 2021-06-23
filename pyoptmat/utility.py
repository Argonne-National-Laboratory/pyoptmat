"""
  Various utility functions used in the tests and examples
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

    Parameters:
      strain:       input strain
      stress_true:  actual stress values
      stress_calc:  simulated stress values

    Additional Parameters:
      alpha:        alpha value for shading
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

    Parameters:
      fn:       function to differentiate via finite differences
      x0:       point at which to take the numerical derivative

    Additional Parameters:
      eps:      perturbation to use
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

    Parameters:
      fn:       function to differentiate via finite differences
      x0:       point at which to take the numerical derivative

    Additional Parameters:
      eps:      perturbation to use
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

def timeseries_interpolate_batch_times(times, values, t):
  """
    Interpolate the time series defined by X to the times defined by t

    Parameters:
      times     input time series as a (ntime,nbatch) array
      values    input value series as a (ntime,nbatch) array
      t         batch times as a (nbatch,) array

    Returns:
      Interpolated values as a (nbatch,) array
  """
  gi = torch.remainder(torch.sum((times - t) <= 0,dim = 0), times.shape[0])
  slopes = (values[gi] - values[gi-1])/(times[gi] - times[gi-1])
  return torch.diagonal(values[gi-1] + slopes * (t - times[gi-1]))

def timeseries_interpolate_single_times(times, values, t):
  """
    Interpolate the time series defined by X to the times defined by t

    Parameters:
      times     input time series as a (ntime,) array
      values    input value series as a (ntime,nbatch) array
      t         times as a scalar

    Returns:
      Interpolated values as a (nbatch,) array
  """
  gi = torch.remainder(torch.sum((times - t) <= 0,dim = 0), times.shape[0])
  slopes = (values[gi] - values[gi-1])/(times[gi,None] - times[gi-1,None])
  return values[gi-1] + slopes * (t - times[gi-1])

def random_parameter(frange):
  """
    Generate a random parameter value as a 1, tensor

    Parameter:
      frange:       range to sample
  """
  if frange is None:
    return nn.Parameter(torch.rand(1))
  else:
    return nn.Parameter(torch.Tensor(1).uniform_(*frange))

def heaviside(X):
  """
    A pytorch-differentiable version of the heaviside function

    Parameters:
      X:        tensor input
  """
  return (torch.sign(X) + 1.0) / 2.0

def macaulay(X):
  """
    A pytorch-differentiable version of the Macualay bracket

    Parameters:
      X:        tensor input
  """
  return X * heaviside(X)
