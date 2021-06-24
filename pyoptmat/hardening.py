"""
  Modules defining isotropic and kinematic hardening models.
"""

import torch
import torch.nn as nn

from pyoptmat import utility

class HardeningModel(nn.Module):
  """
    Superclass for all hardening models.  Right now this does nothing, but
    could be a basis for future expansion.
  """
  def __init__(self):
    super().__init__()

class IsotropicHardeningModel(HardeningModel):
  """
    Superclass for all isotropic hardening models.  Right now this
    does nothing but is here in case we need it in the future.
  """
  def __init__(self):
    super().__init__()

class VoceIsotropicHardeningModel(IsotropicHardeningModel):
  """
    Voce isotropic hardening, defined by
    
    .. math::

      \\sigma_{iso} = h

      \\dot{h} = d (R - h) \\left|\\dot{\\varepsilon}_{in}\\right|

    Args:
      R:                    saturated increase/decrease in flow stress
      d:                    parameter controlling the rate of saturation
      R_scale (optional):   scaling function for R
      d_scale (optional):   scaling function for d
  """
  def __init__(self, R, d, R_scale = lambda x: x, 
      d_scale = lambda x: x):
    super().__init__()
    self.R_param = R
    self.R_scale = R_scale
    self.d_param = d
    self.d_scale = d_scale

    self._setup()

  def _setup(self):
    """
      Setup the model by caching the unscaled parameters
    """
    self.R = self.R_scale(self.R_param)
    self.d = self.d_scale(self.d_param)

  def value(self, h):
    """
      Map from the vector of internal variables to the isotropic hardening
      value

      Args:
        h:      the vector of internal variables for this model
    """
    return h[:,0]

  def dvalue(self, h):
    """
      Derivative of the map with respect to the internal variables

      Args:
        h:      the vector of internal variables for this model
    """
    return torch.ones((h.shape[0],1), device = h.device)

  @property
  def nhist(self):
    """
      The number of internal variables: here just 1
    """
    return 1

  def history_rate(self, s, h, t, ep):
    """
      The rate evolving the internal variables

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.unsqueeze(self.d * (self.R - h[:,0]) * torch.abs(ep), 1)

  def dhistory_rate_dstress(self, s, h, t, ep):
    """
      The derivative of this history rate with respect to the stress

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.zeros_like(h)

  def dhistory_rate_dhistory(self, s, h, t, ep):
    """
      The derivative of the history rate with respect to the internal variables

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.unsqueeze(-torch.unsqueeze(self.d,-1) * 
        torch.ones_like(h) * torch.abs(ep)[:,None], 1)

  def dhistory_rate_derate(self, s, h, t, ep):
    """
      The derivative of the history rate with respect to the inelastic
      strain rate

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.unsqueeze(torch.unsqueeze(self.d * 
      (self.R - h[:,0]) * torch.sign(ep), 1),1)

class KinematicHardeningModel(HardeningModel):
  """
    Common superclass for kinematic hardening models

    Right now this does nothing, but it's available for future expansion
  """
  def __init__(self):
    super().__init__()

class NoKinematicHardeningModel(KinematicHardeningModel):
  """
    The simplest kinematic hardening model: a constant value of 0
  """
  def __init__(self):
    super().__init__()

  def _setup(self):
    """
      Cache the model parameters
    """
    pass

  @property
  def nhist(self):
    """
      The number of internal variables, here 0
    """
    return 0

  def value(self, h):
    """
      The map between the vector of internal variables and the kinematic 
      hardening

      Args:
        h:      vector of internal variables
    """
    return torch.zeros(h.shape[0], device = h.device)

  def dvalue(self, h):
    """
      Derivative of the map to the kinematic hardening with respect to the
      vector of internal variables

      Args:
        h:      vector of internal variables
    """
    return torch.zeros(h.shape[0],0, device = h.device)

  def history_rate(self, s, h, t, ep):
    """
      The history evolution rate.  Here this is an empty vector.

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.empty_like(h)

  def dhistory_rate_dstress(self, s, h, t, ep):
    """
      The derivative of the history rate with respect to the stress.

      Here this is an empty vector.

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.empty_like(h)

  def dhistory_rate_dhistory(self, s, h, t, ep):
    """
      The derivative of the history rate with respect to the history

      Here this is an empty vector.

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.empty(h.shape[0],0,0, device = h.device)

  def dhistory_rate_derate(self, s, h, t, ep):
    """
      The derivative of the history rate with respect to the inelastic
      strain rate.

      Here this is an empty vector.

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.empty(h.shape[0],0,1, device = h.device) 

class FAKinematicHardeningModel(KinematicHardeningModel):
  """
    Frederick and Armstrong hardening, as defined in the (republished) paper:

    Frederick, C. and P. Armstrong. "A mathematical representation of the
    multiaxial Baushcinger effect." Materials at High Temperatures: 24(1)
    pp. 1-26, 2007.

    The kinematic hardening is equal to the single internal variable.

    The variable evolves as:

    .. math::
    
      \\dot{x}=\\frac{2}{3}C\\dot{\\varepsilon}_{in}-gx\\left|\\dot{\\varepsilon}_{in}\\right| 

    Args:
      C:                    kinematic hardening parameter
      g:                    recovery parameter
      C_scale (optional):   scaling function for C
      g_scale (optional):   scaling function for g
  """
  def __init__(self, C, g, C_scale = lambda x: x,
      g_scale = lambda x: x):
    super().__init__()
    self.C_param = C
    self.C_scale = C_scale
    self.g_param = g
    self.g_scale = g_scale

    self._setup()

  def _setup(self):
    """
      Setup the model by precaching the unscaled parameters
    """
    self.C = self.C_scale(self.C_param)
    self.g = self.g_scale(self.g_param)

  def value(self, h):
    """
      The map from the internal variables to the kinematic hardening

      Args:
        h:      vector of internal variables
    """
    return h[:,0]

  def dvalue(self, h):
    """
      Derivative of the map to the kinematic hardening with respect to the
      vector of internal variables

      Args:
        h:      vector of internal variables      
    """
    return torch.ones((h.shape[0],1), device = h.device)

  @property
  def nhist(self):
    """
      The number of internal variables, here just 1
    """
    return 1

  def history_rate(self, s, h, t, ep):
    """
      The evolution rate for the internal variables

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.unsqueeze(
        2.0/3 * self.C * ep - self.g * h[:,0] * torch.abs(ep), 1)

  def dhistory_rate_dstress(self, s, h, t, ep):
    """
      The derivative of the history rate with respect to the stress

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.zeros_like(h)

  def dhistory_rate_dhistory(self, s, h, t, ep):
    """
      The derivative of the history rate with respect to the history

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.unsqueeze(-self.g[...,None] * torch.abs(ep)[:,None], 1)

  def dhistory_rate_derate(self, s, h, t, ep):
    """
      The derivative of the history rate with respect to the inelastic
      strain rate.

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.unsqueeze(torch.unsqueeze(2.0/3 * self.C - 
      self.g * h[:,0] * torch.sign(ep), 1), 1)

class ChabocheHardeningModel(KinematicHardeningModel):
  """
    Chaboche kinematic hardening, as defined by 

    Chaboche, J. and D. Nouailhas. "A unified constitutive modmel for 
    cyclic viscoplasticity and its applications to various stainless steels."
    Journal of Engineering Materials and Technology: 111, pp. 424-430, 1989.

    The model maintains n backstresses and sums them to provide the
    total kinematic hardening
    
    .. math::

      \\sigma_{kin}=\\sum_{i=1}^{n_{kin}}x_{i}

    Each individual backstress evolves per the Frederick-Armstrong model

    .. math::

      \\dot{x}_{i}=\\frac{2}{3}C_{i}\\dot{\\varepsilon}_{in}-g_{i}x_{i}\\left|\\dot{\\varepsilon}_{in}\\right|

    Args:
      C:                    *vector* of hardening coefficients
      g:                    *vector* of recovery coefficients
      C_scale (optional):   scaling function for C
      g_scale (optional):   scaling function for g
  """
  def __init__(self, C, g, C_scale = lambda x: x, g_scale = lambda x: x):
    super().__init__()
    self.C_param = C
    self.C_scale = C_scale
    self.g_param = g
    self.g_scale = g_scale

    self._setup()

    self.nback = self.C.shape[-1]

  def _setup(self):
    """
      Setup the model by caching the unscaled parameter values
    """
    self.C = self.C_scale(self.C_param)
    self.g = self.g_scale(self.g_param)

  def value(self, h):
    """
      The map between the internal variables and the kinematic hardening

      Here :math:`\\sigma_{kin}=\\sum_{i=1}^{n_{kin}}x_{i}`

      Args:
        h:      vector of internal variables
    """
    return torch.sum(h, 1)

  def dvalue(self, h):
    """
      Derivative of the map between the internal variables and the 
      kinematic hardening with respect to the internal variables

      Args:
        h:      vector of internal variables
    """
    return torch.ones((h.shape[0],self.nback), device = h.device)

  @property
  def nhist(self):
    """
      Number of history variables, equal to the number of backstresses
    """
    return self.nback

  def history_rate(self, s, h, t, ep):
    """
      The evolution rate for the internal variables

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return (self.C[None,...] * ep[:,None] - self.g[None,...] * h * 
        torch.abs(ep)[:,None]).reshape(h.shape)

  def dhistory_rate_dstress(self, s, h, t, ep):
    """
      The derivative of the history rate with respect to stress

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.zeros_like(h)

  def dhistory_rate_dhistory(self, s, h, t, ep):
    """
      The derivative of the history rate with respect to the history

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.diag_embed(-self.g[None,...] * 
        torch.abs(ep)[:,None]).reshape(h.shape+h.shape[1:])

  def dhistory_rate_derate(self, s, h, t, ep):
    """
      The derivative of the history rate with respect to the inelastic strain
      rate

      Args:
        s:      stress
        h:      history
        t:      time
        ep:     the inelastic strain rate
    """
    return torch.unsqueeze(self.C[None,...] * torch.ones_like(ep)[:,None] - 
        self.g[None,:] * h * torch.sign(ep)[:,None],-1).reshape(h.shape + (1,))
