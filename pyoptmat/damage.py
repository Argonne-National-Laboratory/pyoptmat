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

  def damage_rate(self, s, d, t):
    """
      The damage rate and the derivative wrt to the damage variable.
      Here it's just zero.

      Parameters:
        s:      stress
        d:      current value of damage
        t:      current time
    """
    return torch.zeros_like(s), torch.zeros_like(s)

  def d_damage_rate_d_s(self, s, d, t):
    """
      Derivative of the damage rate with respect to the stress.

      Here again it's zero

      Parameters:
        s:      stress
        d:      current value of damage
        t:      current time
    """
    return torch.zeros_like(s)

class HayhurstLeckie(DamageModel):
  """
    A Hayhurst-Leckie type damage model, as described in 

    Hayhurst, D. R. and F. A. Leckie. "Constitutive equations for creep
    rutpure." Acta Metallurgica, 25(9): pp. 1059-1070 (1977).

    The model defines the damage rate as


    Parameters:
      A:            Reference stress
      xi:           Stress sensitivity
      phi:          Damage sensitivity

    Additional Parameters:
      A_scale:      scaling function for A
      xi_scale:     scaling function for xi
      phi_scale:    scaling function for phi
  """
  def __init__(self, A, xi, phi, A_scale = lambda x: x,
      xi_scale = lambda x: x, phi_scale = lambda x: x):
    super().__init__()

    self.A_param = A
    self.A_scale = A_scale
    self.xi_param = xi
    self.xi_scale = xi_scale
    self.phi_param = phi
    self.phi_scale = phi_scale

    self._setup()

  def _setup(self):
    """
      Setup the model by precaching the unscaled parameters
    """
    self.A = self.A_scale(self.A_param)
    self.xi = self.xi_scale(self.xi_param)
    self.phi = self.phi_scale(self.phi_param)

  def damage_rate(self, s, d, t):
    """
      Damage rate and the derivative of the rate with respect to the 
      damage variable

      Parameters:
        s:      stress
        d:      damage variable
        t:      time
    """
    return (torch.abs(s)/self.A)**self.xi * (1 - d)**(
        self.xi - self.phi), -(torch.abs(s)/self.A
            )**self.xi * (1 - d)**(self.xi - self.phi - 1)

  def d_damage_rate_d_s(self, s, d, t):
    """
      Derivative of the damage rate with respect to the stress

      Parameters:
        s:      stress
        d:      damage variable
        t:      time
    """
    return (torch.abs(s)/self.A)**(self.xi-1) * (1-d)**(
        self.xi - self.phi) * torch.sign(s)

