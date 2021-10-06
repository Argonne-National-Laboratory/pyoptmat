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

  def dflow_dhist(self, s, h, t, T):
    return torch.zeros(s.shape + (1,) + h.shape[1:])

  def dhist_dstress(self, s, h, t, T):
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
  """
  def __init__(self, n, eta):
    super().__init__()
    self.n = n
    self.eta = eta

  def flow_rate(self, s, h, t, T):
    """
      The uniaxial flow rate itself and the derivative 
      with respect to stress

      Args:
        s:          stress
        h:          history
        t:          time
        T:          temperature
    """
    return ((torch.abs(s)/self.eta(T))**self.n(T) * torch.sign(s),
        self.n(T)*(torch.abs(s)/self.eta(T))**(self.n(T)-1) / self.eta(T))
  
  @property
  def nhist(self):
    """
      The number of internal variables

      Here 0...
    """
    return 0
  
  def history_rate(self, s, h, t, T):
    """
      The history rate for the internal variables

      Here this is zero...

      Args:
        s:      stress
        h:      history
        t:      time
        T:      temperature
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
  """
  def __init__(self, n, eta, s0, isotropic, kinematic):
    super().__init__()
    self.isotropic = isotropic
    self.kinematic = kinematic
    self.n = n
    self.eta = eta
    self.s0 = s0

  def flow_rate(self, s, h, t, T):
    """
      The flow rate itself and the derivative with respect to stress

      Args:
        s:      stress
        h:      internal variables
        t:      time
        T:      temperature
    """
    ih = self.isotropic.value(h[:,:self.isotropic.nhist])
    kh = self.kinematic.value(h[:,self.isotropic.nhist:])

    return (utility.macaulay((torch.abs(s-kh) - self.s0(T) - ih)/self.eta(T))**self.n(T) * torch.sign(s-kh), 
        self.n(T) * utility.macaulay((torch.abs(s-kh) - self.s0(T) -ih)/self.eta(T))**(self.n(T)-1) / self.eta(T))

  def dflow_diso(self, s, h, t, T):
    """
      The derivative of the flow rate with respect to the isotropic hardening

      Args:
        s:      stress
        h:      internal variables
        t:      time
        T:      temperature
    """
    ih = self.isotropic.value(h[:,:self.isotropic.nhist])
    kh = self.kinematic.value(h[:,self.isotropic.nhist:self.isotropic.nhist 
      + self.kinematic.nhist])

    iv = (torch.abs(s-kh) - self.s0(T) - ih)/self.eta(T)

    return -self.n(T) * utility.macaulay(iv)**(self.n(T)-1) / self.eta(T) * torch.sign(s-kh)

  def dflow_dkin(self, s, h, t, T):
    """
      The derivative of the flow rate with respect to the kinematic hardening

      Args:
        s:      stress
        h;      internal variables
        t:      time
        T:      temperature
    """
    ih = self.isotropic.value(h[:,:self.isotropic.nhist])
    kh = self.kinematic.value(h[:,self.isotropic.nhist:self.isotropic.nhist 
      + self.kinematic.nhist])
    
    return -self.n(T) * utility.macaulay((torch.abs(s-kh) - self.s0(T) -ih)/self.eta(T))**(self.n(T)-1) / self.eta(T)

  @property
  def nhist(self):
    """
      The number of internal variables, here the sum from the isotropic
      and kinematic hardening models
    """
    return self.isotropic.nhist + self.kinematic.nhist

  def history_rate(self, s, h, t, T):
    """
      The vector of the rates of the internal variables split into
      portions defined by each hardening model

      The first chunk of entries is for the isotropic hardening,
      the second for the kinematic hardening.

      Args:
        s:      stress
        h;      internal variables
        t:      time
        T:      temperature
    """
    hrate = torch.zeros_like(h)
    hdiv = torch.zeros(h.shape+h.shape[-1:], device = h.device)
    erate, _ = self.flow_rate(s, h, t, T)

    hiso = h[:,:self.isotropic.nhist]
    hkin = h[:,self.isotropic.nhist:]

    hrate[:,:self.isotropic.nhist] = self.isotropic.history_rate(s,
        hiso, t, erate, T)
    hrate[:,self.isotropic.nhist:] = self.kinematic.history_rate(s,
        hkin, t, erate, T)
    
    # History partials
    hdiv[:,:self.isotropic.nhist,:self.isotropic.nhist] = self.isotropic.dhistory_rate_dhistory(s, hiso, t, erate, T)
    hdiv[:,self.isotropic.nhist:,self.isotropic.nhist:] = self.kinematic.dhistory_rate_dhistory(s, hkin, t, erate, T)

    # Strain rate components
    hdiv[:,:self.isotropic.nhist,:self.isotropic.nhist] += torch.matmul(
        self.isotropic.dhistory_rate_derate(s, hiso, t, erate, T),
        torch.matmul(self.dflow_diso(s, h, t, T)[:,None,None],self.isotropic.dvalue(hiso)[:,None,:]))
    hdiv[:,:self.isotropic.nhist,self.isotropic.nhist:] += torch.matmul(
        self.isotropic.dhistory_rate_derate(s, hiso, t, erate, T),
        torch.matmul(self.dflow_dkin(s, h, t, T)[:,None,None],self.kinematic.dvalue(hkin)[:,None,:]))
    hdiv[:,self.isotropic.nhist:,:self.isotropic.nhist] += torch.matmul(
        self.kinematic.dhistory_rate_derate(s, hkin, t, erate, T),
        torch.matmul(self.dflow_diso(s, h, t, T)[:,None,None],self.isotropic.dvalue(hiso)[:,None,:]))
    hdiv[:,self.isotropic.nhist:,self.isotropic.nhist:] += torch.matmul(
        self.kinematic.dhistory_rate_derate(s, hkin, t, erate, T),
        torch.matmul(self.dflow_dkin(s, h, t, T)[:,None,None],self.kinematic.dvalue(hkin)[:,None,:]))

    return hrate, hdiv

  def dflow_dhist(self, s, h, t, T):
    """
      The derivative of the flow rate with respect to the internal variables

      Args:
        s:      stress
        h;      internal variables
        t:      time
        T:      temperature
    """
    res = torch.zeros(s.shape[:1] + (1,) + h.shape[1:], device = h.device)

    erate, _ = self.flow_rate(s, h, t, T)

    hiso = h[:,:self.isotropic.nhist]
    hkin = h[:,self.isotropic.nhist:]

    res[:,0,:self.isotropic.nhist] = self.dflow_diso(s, h, t, T)[:,None] * self.isotropic.dvalue(hiso)
    res[:,0,self.isotropic.nhist:] = self.dflow_dkin(s, h, t, T)[:,None] * self.kinematic.dvalue(hkin)

    return res

  def dhist_dstress(self, s, h, t, T):
    """
      The derivative of the history rate with respect to the stress

      Args:
        s:      stress
        h;      internal variables
        t:      time
        T:      temperature
    """
    res = torch.zeros(h.shape + s.shape[1:], device = h.device)

    erate, derate = self.flow_rate(s, h, t, T)

    hiso = h[:,:self.isotropic.nhist]
    hkin = h[:,self.isotropic.nhist:]


    res[:,:self.isotropic.nhist] = self.isotropic.dhistory_rate_dstress(
        s, hiso, t, erate, T) + self.isotropic.dhistory_rate_derate(
            s, hiso, t, erate, T).bmm(derate[...,None,None])[...,0]
    res[:,self.isotropic.nhist:] = self.kinematic.dhistory_rate_dstress(s,
        hkin, t, erate, T) + self.kinematic.dhistory_rate_derate(s,
            hkin, t, erate, T).bmm(derate[...,None,None])[...,0]

    return res
