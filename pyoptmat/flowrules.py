"""
  Module containing inelastic flow rules
"""

import torch
import torch.nn as nn

from pyoptmat import utility

class FlowRule(nn.Module):
  """
    Superclass of flow rule objects providing the flow_rate

    This implementation provides default zero cross derivatives
  """
  def __init__(self):
    super().__init__()

  def dflow_dhist(self, s, h, t):
    return torch.zeros(s.shape + (1,) + h.shape[1:])

  def dhist_dstress(self, s, h, t):
    return torch.zeros(h.shape + s.shape[1:])

class PerfectViscoplasticity(FlowRule):
  """
    Perfect viscoplasticity defined as

    .. math::

      \\dot{\\varepsilon}_{in}=\\left(\\frac{\\left|\\sigma\\right|}{\\eta}\\right)^{n}\\operatorname{sign}\\left(\\sigma\\right)
      
      \\dot{h} = \\emptyset

    Args:
      n:            rate sensitivity
      eta:          flow viscosity

    Additional Args:
      n_scale:      scaling function mapping parameter -> actual value of n
      eta_scale:    scaling function mapping parameter -> actual value of eta
  """
  def __init__(self, n, eta, n_scale = lambda x: x, eta_scale = lambda x: x):
    super().__init__()
    self.n_param = n
    self.n_scale = n_scale
    self.eta_param = eta
    self.eta_scale = eta_scale
    self._setup()

  def _setup(self):
    """
      Cache the values of the parameters so that we don't need to
      map through the scaling function every time
    """
    self.n = self.n_scale(self.n_param)
    self.eta = self.eta_scale(self.eta_param)

  def flow_rate(self, s, h, t):
    """
      The uniaxial flow rate itself and the derivative 
      with respect to stress

      Args:
        s:          stress
        h:          history
        t:          time
    """
    return ((torch.abs(s)/self.eta)**self.n * torch.sign(s),
        self.n*(torch.abs(s)/self.eta)**(self.n-1) / self.eta)
  
  @property
  def nhist(self):
    """
      The number of internal variables

      Here 0...
    """
    return 0
  
  def history_rate(self, s, h, t):
    """
      The history rate for the internal variables

      Here this is zero...

      Args:
        s:      stress
        h:      history
        t:      time
    """
    return torch.zeros_like(h),torch.zeros(h.shape+h.shape[-1:])

class IsoKinViscoplasticity(FlowRule):
  """
    Viscoplasticity with isotropic and kinematic hardening, defined as

    .. math::

      \\dot{\\varepsilon}_{in}=\\left\\langle \\frac{\\left|\\sigma-x\\right|-s_{0}-k}{\\eta}\\right\\rangle ^{n}\\operatorname{sign}\\left(\\sigma-X\\right)
    
    and where the :py:class:`pyoptmat.hardening.IsotropicHardeningModel` and 
    :py:class:`pyoptmat.hardening.KinematicHardeningModel` objects determine the
    history rate
    
    Args:
      n:                    rate sensitivity
      eta:                  flow viscosity
      s0:                   initial value of flow stress (i.e. "yield stress")
      isotropic:            object providing the isotropic hardening model
      kinematic:            object providing the kinematic hardening model
      n_scale (optional):   scaling function for n
      eta_scale (optional): scaling function for eta
      s0_scale (optional):  scaling function for s0
  """
  def __init__(self, n, eta, s0, isotropic, kinematic,
      n_scale = lambda x: x, eta_scale = lambda x: x, 
      s0_scale = lambda x: x):
    super().__init__()
    self.isotropic = isotropic
    self.kinematic = kinematic
    self.n_param = n
    self.n_scale = n_scale
    self.eta_param = eta
    self.eta_scale = eta_scale
    self.s0_param = s0
    self.s0_scale = s0_scale
    self._setup()

  def _setup(self):
    """
      Setup the model, at least by precaching the unscaled parameters
    """
    self.n = self.n_scale(self.n_param)
    self.eta = self.eta_scale(self.eta_param)
    self.s0 = self.s0_scale(self.s0_param)

    self.isotropic._setup()
    self.kinematic._setup()

  def flow_rate(self, s, h, t):
    """
      The flow rate itself and the derivative with respect to stress

      Args:
        s:      stress
        h:      internal variables
        t:      time
    """
    ih = self.isotropic.value(h[:,:self.isotropic.nhist])
    kh = self.kinematic.value(h[:,self.isotropic.nhist:])

    return (utility.macaulay((torch.abs(s-kh) - self.s0 - ih)/self.eta)**self.n * torch.sign(s-kh), 
        self.n * utility.macaulay((torch.abs(s-kh) - self.s0 -ih)/self.eta)**(self.n-1) / self.eta)

  def dflow_diso(self, s, h, t):
    """
      The derivative of the flow rate with respect to the isotropic hardening

      Args:
        s:      stress
        h:      internal variables
        t:      time
    """
    ih = self.isotropic.value(h[:,:self.isotropic.nhist])
    kh = self.kinematic.value(h[:,self.isotropic.nhist:self.isotropic.nhist 
      + self.kinematic.nhist])

    iv = (torch.abs(s-kh) - self.s0 - ih)/self.eta

    return -self.n * utility.macaulay(iv)**(self.n-1) / self.eta * torch.sign(s-kh)

  def dflow_dkin(self, s, h, t):
    """
      The derivative of the flow rate with respect to the kinematic hardening

      Args:
        s:      stress
        h;      internal variables
        t:      time
    """
    ih = self.isotropic.value(h[:,:self.isotropic.nhist])
    kh = self.kinematic.value(h[:,self.isotropic.nhist:self.isotropic.nhist 
      + self.kinematic.nhist])
    
    return -self.n * utility.macaulay((torch.abs(s-kh) - self.s0 -ih)/self.eta)**(self.n-1) / self.eta

  @property
  def nhist(self):
    """
      The number of internal variables, here the sum from the isotropic
      and kinematic hardening models
    """
    return self.isotropic.nhist + self.kinematic.nhist

  def history_rate(self, s, h, t):
    """
      The vector of the rates of the internal variables split into
      portions defined by each hardening model

      The first chunk of entries is for the isotropic hardening,
      the second for the kinematic hardening.

      Args:
        s:      stress
        h;      internal variables
        t:      time
    """
    hrate = torch.zeros_like(h)
    hdiv = torch.zeros(h.shape+h.shape[-1:], device = h.device)
    erate, _ = self.flow_rate(s, h, t)

    hiso = h[:,:self.isotropic.nhist]
    hkin = h[:,self.isotropic.nhist:]

    hrate[:,:self.isotropic.nhist] = self.isotropic.history_rate(s,
        hiso, t, erate)
    hrate[:,self.isotropic.nhist:] = self.kinematic.history_rate(s,
        hkin, t, erate)
    
    # History partials
    hdiv[:,:self.isotropic.nhist,:self.isotropic.nhist] = self.isotropic.dhistory_rate_dhistory(s, hiso, t, erate)
    hdiv[:,self.isotropic.nhist:,self.isotropic.nhist:] = self.kinematic.dhistory_rate_dhistory(s, hkin, t, erate)

    # Strain rate components
    hdiv[:,:self.isotropic.nhist,:self.isotropic.nhist] += torch.matmul(
        self.isotropic.dhistory_rate_derate(s, hiso, t, erate),
        torch.matmul(self.dflow_diso(s, h, t)[:,None,None],self.isotropic.dvalue(hiso)[:,None,:]))
    hdiv[:,:self.isotropic.nhist,self.isotropic.nhist:] += torch.matmul(
        self.isotropic.dhistory_rate_derate(s, hiso, t, erate),
        torch.matmul(self.dflow_dkin(s, h, t)[:,None,None],self.kinematic.dvalue(hkin)[:,None,:]))
    hdiv[:,self.isotropic.nhist:,:self.isotropic.nhist] += torch.matmul(
        self.kinematic.dhistory_rate_derate(s, hkin, t, erate),
        torch.matmul(self.dflow_diso(s, h, t)[:,None,None],self.isotropic.dvalue(hiso)[:,None,:]))
    hdiv[:,self.isotropic.nhist:,self.isotropic.nhist:] += torch.matmul(
        self.kinematic.dhistory_rate_derate(s, hkin, t, erate),
        torch.matmul(self.dflow_dkin(s, h, t)[:,None,None],self.kinematic.dvalue(hkin)[:,None,:]))

    return hrate, hdiv

  def dflow_dhist(self, s, h, t):
    """
      The derivative of the flow rate with respect to the internal variables

      Args:
        s:      stress
        h;      internal variables
        t:      time
    """
    res = torch.zeros(s.shape[:1] + (1,) + h.shape[1:], device = h.device)

    erate, _ = self.flow_rate(s, h, t)

    hiso = h[:,:self.isotropic.nhist]
    hkin = h[:,self.isotropic.nhist:]

    res[:,0,:self.isotropic.nhist] = self.dflow_diso(s, h, t)[:,None] * self.isotropic.dvalue(hiso)
    res[:,0,self.isotropic.nhist:] = self.dflow_dkin(s, h, t)[:,None] * self.kinematic.dvalue(hkin)

    return res

  def dhist_dstress(self, s, h, t):
    """
      The derivative of the history rate with respect to the stress

      Args:
        s:      stress
        h;      internal variables
        t:      time
    """
    res = torch.zeros(h.shape + s.shape[1:], device = h.device)

    erate, derate = self.flow_rate(s, h, t)

    hiso = h[:,:self.isotropic.nhist]
    hkin = h[:,self.isotropic.nhist:]


    res[:,:self.isotropic.nhist] = self.isotropic.dhistory_rate_dstress(
        s, hiso, t, erate) + self.isotropic.dhistory_rate_derate(
            s, hiso, t, erate).bmm(derate[...,None,None])[...,0]
    res[:,self.isotropic.nhist:] = self.kinematic.dhistory_rate_dstress(s,
        hkin, t, erate) + self.kinematic.dhistory_rate_derate(s,
            hkin, t, erate).bmm(derate[...,None,None])[...,0]

    return res
