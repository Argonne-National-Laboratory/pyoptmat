"""
  Various damage models, which can be tacked onto a InelasticModel to
  degrade the material response
"""
import torch
import torch.nn as nn

class DamageModel(nn.Module):
  """
    Superclass for damage models that modify the viscoplastic flow rate
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

class NoDamage(DamageModel):
  """
    The default damage model, it doesn't actually alter the flow rate
    in any way.
  """
  def __init__(self):
    super().__init__()

  def _setup(self):
    """
      Cache parameter values, but here there aren't any
    """
    pass

  def damage_rate(self, s, d, t, T):
    """
      The damage rate and the derivative wrt to the damage variable.
      Here it's just zero.

      Args:
        s:      stress
        d:      current value of damage
        t:      current time
        T:      current temperature
    """
    return torch.zeros_like(s), torch.zeros_like(s)

  def d_damage_rate_d_s(self, s, d, t, T):
    """
      Derivative of the damage rate with respect to the stress.

      Here again it's zero

      Args:
        s:      stress
        d:      current value of damage
        t:      current time
        T:      current temperature
    """
    return torch.zeros_like(s)

class HayhurstLeckie(DamageModel):
  """
    A Hayhurst-Leckie type damage model, as described in 

    Hayhurst, D. R. and F. A. Leckie. "Constitutive equations for creep
    rutpure." Acta Metallurgica, 25(9): pp. 1059-1070 (1977).

    The model defines the damage rate as
    
    .. math::

      \\left(\\frac{\\left|\\sigma\\right|}{A}\\right)^{\\xi}\\left(1-d\\right)^{\\xi-\\phi}

    Args:
      A:                    Reference stress
      xi:                   Stress sensitivity
      phi:                  Damage sensitivity
  """
  def __init__(self, A, xi, phi):
    super().__init__()

    self.A = A
    self.xi = xi
    self.phi = phi

  def damage_rate(self, s, d, t, T):
    """
      Damage rate and the derivative of the rate with respect to the 
      damage variable

      Args:
        s:      stress
        d:      damage variable
        t:      time
        T:      temperature
    """
    return (torch.abs(s)/self.A(T))**self.xi(T) * (1 - d)**(
        self.xi(T) - self.phi(T)), -(torch.abs(s)/self.A(T)
            )**self.xi(T) * (1 - d)**(self.xi(T) - self.phi(T) - 1)

  def d_damage_rate_d_s(self, s, d, t, T):
    """
      Derivative of the damage rate with respect to the stress

      Args:
        s:      stress
        d:      damage variable
        t:      time
        T:      temperature
    """
    return (torch.abs(s)/self.A(T))**(self.xi(T)-1) * (1-d)**(
        self.xi(T) - self.phi(T)) * torch.sign(s)

