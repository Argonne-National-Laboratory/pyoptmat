# pylint: disable=abstract-method, useless-super-delegation
"""
  Various damage models, which can be tacked onto a InelasticModel to
  degrade the material response over time or accumulated strain.

  These are standard continuum damage mechanics models in the line of
  :cite:`chaboche1988continuum`.
"""
import torch
from torch import nn


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

    def damage_rate(self, s, d, t, T, e):
        """
        The damage rate and the derivative wrt to the damage variable.
        Here it's just zero.

        Args:
          s (torch.tensor):      stress
          d (torch.tensor):      current value of damage
          t (torch.tensor):      current time
          T (torch.tensor):      current temperature
          e (torch.tensor):      total strain rate
        """
        return torch.zeros_like(s), torch.zeros_like(s)

    def d_damage_rate_d_s(self, s, d, t, T, e):
        """
        Derivative of the damage rate with respect to the stress.

        Here again it's zero

        Args:
          s (torch.tensor):      stress
          d (torch.tensor):      current value of damage
          t (torch.tensor):      current time
          T (torch.tensor):      current temperature
          e (torch.tensor):      total strain rate
        """
        return torch.zeros_like(s)

    def d_damage_rate_d_e(self, s, d, t, T, e):
        """
        Derivative of the damage rate with respect to the strain rate

        Here again it's zero

        Args:
          s (torch.tensor):      stress
          d (torch.tensor):      current value of damage
          t (torch.tensor):      current time
          T (torch.tensor):      current temperature
          e (torch.tensor):      total strain rate
        """
        return torch.zeros_like(e)


class HayhurstLeckie(DamageModel):
    """
    A Hayhurst-Leckie type damage model, as described in
    :cite:`leckie1977constitutive`

    The model defines the damage rate as

    .. math::

      \\left(\\frac{\\left|\\sigma\\right|}{A}\\right)^{\\xi}\\left(1-d\\right)^{\\xi-\\phi}

    Args:
      A (torch.tensor):     Reference stress
      xi (torch.tensor):    Stress sensitivity
      phi (torch.tensor):   Damage sensitivity
    """

    def __init__(self, A, xi, phi):
        super().__init__()

        self.A = A
        self.xi = xi
        self.phi = phi

    def damage_rate(self, s, d, t, T, e):
        """
        Damage rate and the derivative of the rate with respect to the
        damage variable

        Args:
          s (torch.tensor):      stress
          d (torch.tensor):      damage variable
          t (torch.tensor):      time
          T (torch.tensor):      temperature
          e (torch.tensor):      total strain rate
        """
        return (torch.abs(s) / self.A(T)) ** self.xi(T) * (1 - d) ** (
            self.xi(T) - self.phi(T)
        ), -((torch.abs(s) / self.A(T)) ** self.xi(T)) * (1 - d) ** (
            self.xi(T) - self.phi(T) - 1
        )

    def d_damage_rate_d_s(self, s, d, t, T, e):
        """
        Derivative of the damage rate with respect to the stress

        Args:
          s (torch.tensor):      stress
          d (torch.tensor):      damage variable
          t (torch.tensor):      time
          T (torch.tensor):      temperature
        """
        return (
            (torch.abs(s) / self.A(T)) ** (self.xi(T) - 1)
            * (1 - d) ** (self.xi(T) - self.phi(T))
            * torch.sign(s)
        )

    def d_damage_rate_d_e(self, s, d, t, T, e):
        """
        Derivative of the damage rate with respect to the strain rate

        Here again it's zero

        Args:
          s (torch.tensor):      stress
          d (torch.tensor):      current value of damage
          t (torch.tensor):      current time
          T (torch.tensor):      current temperature
          e (torch.tensor):      total strain rate
        """
        return torch.zeros_like(e)
