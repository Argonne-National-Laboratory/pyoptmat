# pylint: disable=abstract-method, useless-super-delegation, too-many-lines, duplicate-code

"""
  Modules defining isotropic and kinematic hardening models.

  These models provide:

  1. A set of internal variables
  2. The evolution equations defining each of those variables
  3. A map between the internal variables and the actual value of
     isotropic/kinematic hardening used in the
     :py:class:`pyoptmat.flowrules.FlowRule`
  4. The derivative of that map with respect to the internal variables
"""

import numpy as np

import torch
from torch import nn

from pyoptmat import temperature


class HardeningModel(nn.Module):
    """
    Superclass for all hardening models.  Right now this does nothing, but
    could be a basis for future expansion.
    """

    def __init__(self):
        super().__init__()

    def dhistory_rate_dtotalrate(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the total
        strain rate

        This will be zero in most models

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to the total rate
        """
        return torch.zeros(h.shape, device=h.device)


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
      R (|TP|): saturated increase/decrease in flow stress
      d (|TP|): parameter controlling the rate of saturation
    """

    def __init__(self, R, d):
        super().__init__()
        self.R = R
        self.d = d

    def value(self, h):
        """
        Map from the vector of internal variables to the isotropic hardening
        value

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the isotropic hardening value
        """
        return h[..., 0]

    def dvalue(self, h):
        """
        Derivative of the map with respect to the internal variables

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the derivative of the isotropic hardening value
                              with respect to the internal variables
        """
        return torch.ones(h.shape[:-1] + (1,), device=h.device)

    @property
    def nhist(self):
        """
        The number of internal variables: here just 1
        """
        return 1

    def history_rate(self, s, h, t, ep, T, e):
        """
        The rate evolving the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       internal variable rate
        """
        return torch.unsqueeze(self.d(T) * (self.R(T) - h[..., 0]) * torch.abs(ep), -1)

    def dhistory_rate_dstress(self, s, h, t, ep, T, e):
        """
        The derivative of this history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to stress
        """
        return torch.zeros_like(h)

    def dhistory_rate_dhistory(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to history
        """
        return (-self.d(T) * torch.ones_like(h[..., 0]) * torch.abs(ep))[
            ..., None, None
        ]

    def dhistory_rate_derate(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the inelastic
        strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to the inelastic rate
        """
        return (self.d(T) * (self.R(T) - h[..., 0]) * torch.sign(ep))[..., None, None]


class Theta0VoceIsotropicHardeningModel(IsotropicHardeningModel):
    """
    Reparameterized Voce isotropic hardening, defined by

    .. math::

      \\sigma_{iso} = h

      \\dot{h} = \\theta_0 (1-h/\\tau) \\left|\\dot{\\varepsilon}_{in}\\right|

    This gives the same response as :py:class:`pyoptmat.hardening.VoceIsotropicHardeningModel`
    it just uses a different definition of the parameters

    Args:
      tau (|TP|):   saturated increase/decrease in flow stress
      theta (|TP|): initial hardening rate
    """

    def __init__(self, tau, theta):
        super().__init__()
        self.tau = tau
        self.theta = theta

    def value(self, h):
        """
        Map from the vector of internal variables to the isotropic hardening
        value

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the isotropic hardening value
        """
        return h[..., 0]

    def dvalue(self, h):
        """
        Map from the vector of internal variables to the isotropic hardening
        value

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the isotropic hardening value
        """
        return torch.ones(h.shape[:-1] + (1,), device=h.device)

    @property
    def nhist(self):
        """
        The number of internal variables: here just 1
        """
        return 1

    def history_rate(self, s, h, t, ep, T, e):
        """
        The rate evolving the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       internal variable rate
        """
        return torch.unsqueeze(
            self.theta(T) * (1.0 - h[..., 0] / self.tau(T)) * torch.abs(ep), -1
        )

    def dhistory_rate_dstress(self, s, h, t, ep, T, e):
        """
        The derivative of this history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to stress
        """
        return torch.zeros_like(h)

    def dhistory_rate_dhistory(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to history
        """
        return (
            -self.theta(T) / self.tau(T) * torch.ones_like(h[..., 0]) * torch.abs(ep)
        )[..., None, None]

    def dhistory_rate_derate(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the inelastic
        strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to the inelastic rate
        """
        return (self.theta(T) * (1.0 - h[..., 0] / self.tau(T)) * torch.sign(ep))[
            ..., None, None
        ]


class Theta0RecoveryVoceIsotropicHardeningModel(IsotropicHardeningModel):
    # pylint: disable=line-too-long
    """
    Voce isotropic hardening with static recovery, defined by

    .. math::

      \\sigma_{iso} = h

      \\dot{h} = \\theta_0 \\left(1-\\frac{h}{\\tau}\\right) \\left|\\dot{\\varepsilon}_{in}\\right| + r_1 \\left(R_0 - h\\right) \\left| R_0 - h \\right|^{r_2 - 1}

    Args:
      tau (|TP|):       saturated increase/decrease in flow stress
      theta (|TP|):     initial hardening rate
      R0 (|TP|):        static recovery threshold
      r1 (|TP|):        static recovery prefactor
      r2 (|TP|):        static recovery exponent
    """

    def __init__(self, tau, theta, R0, r1, r2):
        super().__init__()
        self.tau = tau
        self.theta = theta
        self.R0 = R0
        self.r1 = r1
        self.r2 = r2

    def value(self, h):
        """
        Map from the vector of internal variables to the isotropic hardening
        value

        Args:
          h:      the vector of internal variables for this model
        """
        return h[..., 0]

    def dvalue(self, h):
        """
        Derivative of the map with respect to the internal variables

        Args:
          h:      the vector of internal variables for this model
        """
        return torch.ones(h.shape[:-1] + (1,), device=h.device)

    @property
    def nhist(self):
        """
        The number of internal variables: here just 1
        """
        return 1

    def history_rate(self, s, h, t, ep, T, e):
        """
        The rate evolving the internal variables

        Args:
          s:      stress
          h:      history
          t:      time
          ep:     the inelastic strain rate
          T:      the temperature
          e (torch.tensor):   total strain rate
        """
        return torch.unsqueeze(
            self.theta(T) * (1.0 - h[..., 0] / self.tau(T)) * torch.abs(ep)
            + self.r1(T)
            * (self.R0(T) - h[..., 0])
            * torch.abs(self.R0(T) - h[..., 0]) ** (self.r2(T) - 1.0),
            -1,
        )

    def dhistory_rate_dstress(self, s, h, t, ep, T, e):
        """
        The derivative of this history rate with respect to the stress

        Args:
          s:      stress
          h:      history
          t:      time
          ep:     the inelastic strain rate
          T:      temperature
          e (torch.tensor):   total strain rate
        """
        return torch.zeros_like(h)

    def dhistory_rate_dhistory(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the internal variables

        Args:
          s:      stress
          h:      history
          t:      time
          ep:     the inelastic strain rate
          T:      temperature
          e (torch.tensor):   total strain rate
        """
        recovery = (
            self.r2(T)
            * self.r1(T)
            * torch.abs(self.R0(T) - h[..., 0]) ** (self.r2(T) - 1.0)
        )[..., None, None]
        return (
            -self.theta(T) / self.tau(T) * torch.ones_like(h[..., 0]) * torch.abs(ep)
        )[..., None, None] - recovery

    def dhistory_rate_derate(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the inelastic
        strain rate

        Args:
          s:      stress
          h:      history
          t:      time
          ep:     the inelastic strain rate
          T:      temperature
          e (torch.tensor):   total strain rate
        """
        return (self.theta(T) * (1.0 - h[..., 0] / self.tau(T)) * torch.sign(ep))[
            ..., None, None
        ]


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
        return torch.zeros(h.shape[:-1], device=h.device)

    def dvalue(self, h):
        """
        Derivative of the map to the kinematic hardening with respect to the
        vector of internal variables

        Args:
          h:      vector of internal variables
        """
        return torch.zeros(h.shape[:-1] + (0,), device=h.device)

    def history_rate(self, s, h, t, ep, T, e):
        """
        The history evolution rate.  Here this is an empty vector.

        Args:
          s:      stress
          h:      history
          t:      time
          ep:     the inelastic strain rate
          T:      the temperature
          e (torch.tensor):   total strain rate
        """
        return torch.empty_like(h)

    def dhistory_rate_dstress(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the stress.

        Here this is an empty vector.

        Args:
          s:      stress
          h:      history
          t:      time
          ep:     the inelastic strain rate
          T:      temperature
          e (torch.tensor):   total strain rate
        """
        return torch.empty_like(h)

    def dhistory_rate_dhistory(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the history

        Here this is an empty vector.

        Args:
          s:      stress
          h:      history
          t:      time
          ep:     the inelastic strain rate
          T:      temperature
          e (torch.tensor):   total strain rate
        """
        return torch.empty(h.shape[:-1] + (0, 0), device=h.device)

    def dhistory_rate_derate(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the inelastic
        strain rate.

        Here this is an empty vector.

        Args:
          s:      stress
          h:      history
          t:      time
          ep:     the inelastic strain rate
          T:      temperature
          e (torch.tensor):   total strain rate
        """
        return torch.empty(h.shape[:-1] + (0, 1), device=h.device)


class FAKinematicHardeningModel(KinematicHardeningModel):
    # pylint: disable=line-too-long
    """
    Frederick and Armstrong hardening, as defined in :cite:`frederick2007mathematical`

    The kinematic hardening is equal to the single internal variable.

    The variable evolves as:

    .. math::

      \\dot{x}=\\frac{2}{3}C\\dot{\\varepsilon}_{in}-gx\\left|\\dot{\\varepsilon}_{in}\\right| - b\\left| h \\right|^{r-1} h

    where the static recovery defaults to zero

    Args:
      C (|TP|):     kinematic hardening parameter
      g (|TP|):     recovery parameter
      b (optional): static recovery prefactor
      r (optional): static recovery exponent
    """

    def __init__(self, C, g, b=None, r=None):
        super().__init__()
        self.C = C
        self.g = g

        Cdev = self.C.device

        if b is None:
            b = temperature.ConstantParameter(torch.zeros(self.C.shape, device=Cdev))
        if r is None:
            r = temperature.ConstantParameter(torch.ones(self.C.shape, device=Cdev))

        self.b = b
        self.r = r

    def value(self, h):
        """
        Map from the vector of internal variables to the kinematic hardening
        value

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the kinematic hardening value
        """
        return h[..., 0]

    def dvalue(self, h):
        """
        Derivative of the map with respect to the internal variables

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the derivative of the kinematic hardening value
                              with respect to the internal variables
        """
        return torch.ones(h.shape[:-1] + (1,), device=h.device)

    @property
    def nhist(self):
        """
        The number of internal variables, here just 1
        """
        return 1

    def history_rate(self, s, h, t, ep, T, e):
        """
        The rate evolving the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       internal variable rate
        """
        return torch.unsqueeze(
            self.C(T) * torch.ones_like(h[..., 0]) * ep
            - self.g(T) * h[..., 0] * torch.abs(ep)
            - self.b(T) * torch.abs(h[..., 0]) ** (self.r(T) - 1.0) * h[..., 0],
            -1,
        )

    def dhistory_rate_dstress(self, s, h, t, ep, T, e):
        """
        The derivative of this history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to stress
        """
        return torch.zeros_like(h)

    def dhistory_rate_dhistory(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to history
        """
        return (
            -self.g(T) * torch.ones_like(h[..., 0]) * torch.abs(ep)
            - (self.b(T) * self.r(T) * torch.abs(h[..., 0]) ** (self.r(T) - 1.0))
        )[..., None, None]

    def dhistory_rate_derate(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the inelastic
        strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to the inelastic rate
        """
        return (self.C(T) - self.g(T) * h[..., 0] * torch.sign(ep))[..., None, None]


class ChabocheHardeningModel(KinematicHardeningModel):
    # pylint: disable=line-too-long
    """
    Chaboche kinematic hardening, as defined in :cite:`chaboche1989unified`

    This version does *not* include static recovery

    The model maintains :math:`n` backstresses and sums them to provide the
    total kinematic hardening

    .. math::

      \\sigma_{kin}=\\sum_{i=1}^{n_{kin}}x_{i}

    Each individual backstress evolves per the Frederick-Armstrong model

    .. math::

      \\dot{x}_{i}=\\frac{2}{3}C_{i}\\dot{\\varepsilon}_{in}-g_{i}x_{i}\\left|\\dot{\\varepsilon}_{in}\\right|

    Args:
      C (list of |TP|): *vector* of hardening coefficients
      g (list of |TP|): *vector* of recovery coefficients
    """

    def __init__(self, C, g):
        super().__init__()
        self.C = C
        self.g = g

        self.nback = self.C.shape[-1]

    def value(self, h):
        """
        Map from the vector of internal variables to the kinematic hardening
        value

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the kinematic hardening value
        """
        return torch.sum(h, -1)

    def dvalue(self, h):
        """
        Derivative of the map with respect to the internal variables

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the derivative of the kinematic hardening value
                              with respect to the internal variables
        """
        return torch.ones(h.shape[:-1] + (self.nback,), device=h.device)

    @property
    def nhist(self):
        """
        Number of history variables, equal to the number of backstresses
        """
        return self.nback

    def history_rate(self, s, h, t, ep, T, e):
        """
        The rate evolving the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       internal variable rate
        """
        return (
            self.C(T)[None, ...] * ep[..., None]
            - self.g(T)[None, ...] * h * torch.abs(ep)[..., None]
        ).reshape(h.shape)

    def dhistory_rate_dstress(self, s, h, t, ep, T, e):
        """
        The derivative of this history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to stress
        """
        return torch.zeros_like(h)

    def dhistory_rate_dhistory(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to history
        """
        return torch.diag_embed(
            -self.g(T)[None, ...] * torch.abs(ep)[..., None]
        ).reshape(h.shape + (self.nback,))

    def dhistory_rate_derate(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the inelastic
        strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to the inelastic rate
        """
        return torch.unsqueeze(
            self.C(T)[None, ...] * torch.ones_like(ep)[..., None]
            - self.g(T)[None, ...] * h * torch.sign(ep)[..., None],
            -1,
        ).reshape(h.shape + (1,))


class ChabocheHardeningModelRecovery(KinematicHardeningModel):
    # pylint: disable=line-too-long
    """
    Chaboche kinematic hardening, as defined in :cite:`chaboche1989unified`

    This version *does* include static recovery

    The model maintains :math:`n` backstresses and sums them to provide the
    total kinematic hardening

    .. math::

      \\sigma_{kin}=\\sum_{i=1}^{n_{kin}}x_{i}

    Each individual backstress evolves per the Frederick-Armstrong model

    .. math::

      \\dot{x}_{i}=\\frac{2}{3}C_{i}\\dot{\\varepsilon}_{in}-g_{i}x_{i}\\left|\\dot{\\varepsilon}_{in}\\right| - b\\left| h \\right|^{r-1} h

    .. math::

      \\sigma_{kin}=\\sum_{i=1}^{n_{kin}}x_{i}

    Args:
      C (list of |TP|): *vector* of hardening coefficients
      g (list of |TP|): *vector* of recovery coefficients
      b (list of |TP|): *vector* of static recovery prefactors
      r (list of |TP|): *vector* of static recovery exponents
    """

    def __init__(self, C, g, b, r):
        super().__init__()
        self.C = C
        self.g = g
        self.b = b
        self.r = r

        self.nback = self.C.shape[-1]

    def value(self, h):
        """
        Map from the vector of internal variables to the kinematic hardening
        value

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the kinematic hardening value
        """
        return torch.sum(h, -1)

    def dvalue(self, h):
        """
        Derivative of the map with respect to the internal variables

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the derivative of the kinematic hardening value
                              with respect to the internal variables
        """
        return torch.ones(h.shape[:-1] + (self.nback,), device=h.device)

    @property
    def nhist(self):
        """
        Number of history variables, equal to the number of backstresses
        """
        return self.nback

    def history_rate(self, s, h, t, ep, T, e):
        """
        The rate evolving the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       internal variable rate
        """
        return (
            self.C(T)[None, ...] * ep[..., None]
            - self.g(T)[None, ...] * h * torch.abs(ep)[..., None]
            - self.b(T)[None, ...] * torch.abs(h) ** (self.r(T)[None, ...] - 1.0) * h
        ).reshape(h.shape)

    def dhistory_rate_dstress(self, s, h, t, ep, T, e):
        """
        The derivative of this history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to stress
        """
        return torch.zeros_like(h)

    def dhistory_rate_dhistory(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to history
        """
        return torch.diag_embed(
            -self.g(T)[None, ...] * torch.abs(ep)[..., None]
        ).reshape(h.shape + (self.nback,)) + torch.diag_embed(
            -self.b(T)[None, ...]
            * self.r(T)[None, ...]
            * torch.abs(h) ** (self.r(T)[None, ...] - 1.0)
        ).reshape(
            h.shape + (self.nback,)
        )

    def dhistory_rate_derate(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the inelastic
        strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to the inelastic rate
        """
        return torch.unsqueeze(
            self.C(T)[None, ...] * torch.ones_like(ep)[..., None]
            - self.g(T)[None, ...] * h * torch.sign(ep)[..., None],
            -1,
        ).reshape(h.shape + (1,))


class SuperimposedKinematicHardening(KinematicHardeningModel):
    # pylint: disable=line-too-long
    """
    Sum the contributions of several kinematic hardening models

    Args:
      models (list of models):   list of KinematicHardening models
    """

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

        self.nmodels = len(self.models)
        self.nhist_per = [m.nhist for m in self.models]
        self.offsets = [0] + list(np.cumsum(self.nhist_per))[:-1]

    def value(self, h):
        """
        Map from the vector of internal variables to the kinematic hardening
        value

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the kinematic hardening value
        """
        v = torch.zeros(h.shape[:-1], device=h.device)
        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            v += model.value(h[..., o : o + n])

        return v

    def dvalue(self, h):
        """
        Derivative of the map with respect to the internal variables

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the derivative of the kinematic hardening value
                              with respect to the internal variables
        """
        dv = torch.zeros(h.shape[:-1] + (self.nhist,), device=h.device)
        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            dv[..., o : o + n] = model.dvalue(h[..., o : o + n])

        return dv

    @property
    def nhist(self):
        """
        Number of history variables
        """
        return sum(self.nhist_per)

    def history_rate(self, s, h, t, ep, T, e):
        """
        The rate evolving the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       internal variable rate
        """
        hr = torch.zeros_like(h)
        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            hr[..., o : o + n] = model.history_rate(s, h[..., o : o + n], t, ep, T, e)

        return hr

    def dhistory_rate_dstress(self, s, h, t, ep, T, e):
        """
        The derivative of this history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to stress
        """
        dhr = torch.zeros_like(h)
        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            dhr[..., o : o + n] = model.dhistory_rate_dstress(
                s, h[..., o : o + n], t, ep, T, e
            )

        return dhr

    def dhistory_rate_dhistory(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to history
        """
        dhr = torch.zeros(h.shape[:-1] + (self.nhist, self.nhist), device=s.device)
        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            dhr[..., o : o + n, o : o + n] = model.dhistory_rate_dhistory(
                s, h[..., o : o + n], t, ep, T, e
            )

        return dhr

    def dhistory_rate_derate(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the inelastic
        strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to the inelastic rate
        """
        dhr = torch.zeros(h.shape + (1,), device=s.device)
        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            dhr[..., o : o + n, 0:1] = model.dhistory_rate_derate(
                s, h[..., o : o + n], t, ep, T, e
            )

        return dhr

    def dhistory_rate_dtotalrate(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the total
        strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to the total rate
        """
        dhr = torch.zeros(h.shape, device=s.device)
        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            dhr[..., o : o + n] = model.dhistory_rate_dtotalrate(
                s, h[..., o : o + n], t, ep, T, e
            )

        return dhr
