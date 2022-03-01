# pylint: disable=abstract-method, no-self-use, useless-super-delegation, line-too-long

"""
  Module containing inelastic flow rules.  These provide the rate of the
  viscoplastic strain and internal variables as a function of the stress,
  the current values of the internal variables, the time, and the temperature.

  So these objects define two functions

    .. math::

      \\sigma, h, t, T \\rightarrow \\dot{\\varepsilon}_{in}

      \\sigma, h, t, T \\rightarrow \\dot{h}

  In addition, the object needs to define the derivative of the inelastic
  strain rate and internal variable evolution rates with respect to the
  current values of stress and the current values of the internal variables.
  The objects return "self" derivatives (i.e. the derivative of the inelastic
  strain rate with respect to stress and the derivative of the internal
  variable rate with respect to the internal variables) along with the rates
  themselves.  The "cross" derivatives are defined with separate methods.
"""

import torch
from torch import nn

from pyoptmat import utility


class FlowRule(nn.Module):
    """
    Superclass for flow rule models

    This implementation provides default zero cross derivatives and that's it.
    """

    def __init__(self):
        super().__init__()

    def dflow_dhist(self, s, h, t, T):
        """
        The derivative of the flow rate with respect to the internal variables

        The superclass implementation provides a default of zero with the
        right shape.

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature

        Returns:
          torch.tensor:       derivative of flow rate with respect to the
                              internal variables
        """
        return torch.zeros(s.shape + (1,) + h.shape[1:])

    def dhist_dstress(self, s, h, t, T):
        """
        The derivative of the flow rate with respect to the stress

        The superclass implementation provides a default of zero with the
        right shape.

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature

        Returns:
          torch.tensor:       derivative of flow rate with respect to the stress
        """
        return torch.zeros(h.shape + s.shape[1:])


class PerfectViscoplasticity(FlowRule):
    """
    Perfect viscoplasticity defined as

    .. math::

      \\dot{\\varepsilon}_{in}=\\left(\\frac{\\left|\\sigma\\right|}{\\eta}\\right)^{n}\\operatorname{sign}\\left(\\sigma\\right)

      \\dot{h} = \\emptyset

    Args:
      n (|TP|):     rate sensitivity
      eta (|TP|):   flow viscosity
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
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature

        Returns:
          tuple(torch.tensor, torch.tensor):    the flow rate and the derivative
                                                of the flow rate with
                                                respect to stress
        """
        return (
            (torch.abs(s) / self.eta(T)) ** self.n(T) * torch.sign(s),
            self.n(T) * (torch.abs(s) / self.eta(T)) ** (self.n(T) - 1) / self.eta(T),
        )

    @property
    def nhist(self):
        """
        The number of internal variables

        Here 0...
        """
        return 0

    def history_rate(self, s, h, t, T):
        """
        The history rate and the derivative of the history rate with respect
        to the current history

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature

        Returns:
          tuple(torch.tensor, torch.tensor):    the history rate and the
                                                derivative of the history rate
                                                with respect to history
        """
        return torch.zeros_like(h), torch.zeros(h.shape + h.shape[-1:])


class IsoKinViscoplasticity(FlowRule):
    """
    Viscoplasticity with isotropic and kinematic hardening, defined as

    .. math::

      \\dot{\\varepsilon}_{in}=\\left\\langle \\frac{\\left|\\sigma-x\\right|-s_{0}-k}{\\eta}\\right\\rangle ^{n}\\operatorname{sign}\\left(\\sigma-X\\right)

    and where the :py:class:`pyoptmat.hardening.IsotropicHardeningModel` and
    :py:class:`pyoptmat.hardening.KinematicHardeningModel` objects determine the
    history rate.

    The :py:class:`pyoptmat.hardening.IsotropicHardeningModel` and
    :py:class:`pyoptmat.hardening.KinematicHardeningModel` objects each define both a
    set of internal variables, including the corresponding rate forms and Jacobians,
    but also a map from those internal variables to the isotropic hardening
    value :math:`k` (for :py:class:`pyoptmat.hardening.IsotropicHardeningModel`)
    and the kinematic hardening value :math:`x`
    (for :py:class:`pyoptmat.hardening.KinematicHardeningModel`), along with
    the derivatives of those maps.  All this information is required to assemble
    the information this class needs to provide.

    Args:
      n (|TP|):         rate sensitivity
      eta (|TP|):       flow viscosity
      s0 (|TP|):        initial value of flow stress (i.e. the threshold stress)
      isotropic (:py:class:`pyoptmat.hardening.IsotropicHardeningModel`): object providing the isotropic hardening model
      kinematic (:py:class:`pyoptmat.hardening.IsotropicHardeningModel`): object providing the kinematic hardening model
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
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature

        Returns:
          tuple(torch.tensor, torch.tensor):  the flow rate and the derivative of the flow rate with
                                              respect to stress
        """
        ih = self.isotropic.value(h[:, : self.isotropic.nhist])
        kh = self.kinematic.value(h[:, self.isotropic.nhist :])

        return (
            utility.macaulay((torch.abs(s - kh) - self.s0(T) - ih) / self.eta(T))
            ** self.n(T)
            * torch.sign(s - kh),
            self.n(T)
            * utility.macaulay((torch.abs(s - kh) - self.s0(T) - ih) / self.eta(T))
            ** (self.n(T) - 1)
            / self.eta(T),
        )

    def dflow_diso(self, s, h, t, T):
        """
        The derivative of the flow rate with respect to the isotropic hardening

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        ih = self.isotropic.value(h[:, : self.isotropic.nhist])
        kh = self.kinematic.value(
            h[:, self.isotropic.nhist : self.isotropic.nhist + self.kinematic.nhist]
        )

        iv = (torch.abs(s - kh) - self.s0(T) - ih) / self.eta(T)

        return (
            -self.n(T)
            * utility.macaulay(iv) ** (self.n(T) - 1)
            / self.eta(T)
            * torch.sign(s - kh)
        )

    def dflow_dkin(self, s, h, t, T):
        """
        The derivative of the flow rate with respect to the kinematic hardening

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        ih = self.isotropic.value(h[:, : self.isotropic.nhist])
        kh = self.kinematic.value(
            h[:, self.isotropic.nhist : self.isotropic.nhist + self.kinematic.nhist]
        )

        return (
            -self.n(T)
            * utility.macaulay((torch.abs(s - kh) - self.s0(T) - ih) / self.eta(T))
            ** (self.n(T) - 1)
            / self.eta(T)
        )

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
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature

        Returns:
          tuple(torch.tensor, torch.tensor):    the history rate and the
                                                derivative of the history rate
                                                with respect to history
        """
        hrate = torch.zeros_like(h)
        hdiv = torch.zeros(h.shape + h.shape[-1:], device=h.device)
        erate, _ = self.flow_rate(s, h, t, T)

        hiso = h[:, : self.isotropic.nhist]
        hkin = h[:, self.isotropic.nhist :]

        hrate[:, : self.isotropic.nhist] = self.isotropic.history_rate(
            s, hiso, t, erate, T
        )
        hrate[:, self.isotropic.nhist :] = self.kinematic.history_rate(
            s, hkin, t, erate, T
        )

        # History partials
        hdiv[
            :, : self.isotropic.nhist, : self.isotropic.nhist
        ] = self.isotropic.dhistory_rate_dhistory(s, hiso, t, erate, T)
        hdiv[
            :, self.isotropic.nhist :, self.isotropic.nhist :
        ] = self.kinematic.dhistory_rate_dhistory(s, hkin, t, erate, T)

        # Strain rate components
        hdiv[:, : self.isotropic.nhist, : self.isotropic.nhist] += torch.matmul(
            self.isotropic.dhistory_rate_derate(s, hiso, t, erate, T),
            torch.matmul(
                self.dflow_diso(s, h, t, T)[:, None, None],
                self.isotropic.dvalue(hiso)[:, None, :],
            ),
        )
        hdiv[:, : self.isotropic.nhist, self.isotropic.nhist :] += torch.matmul(
            self.isotropic.dhistory_rate_derate(s, hiso, t, erate, T),
            torch.matmul(
                self.dflow_dkin(s, h, t, T)[:, None, None],
                self.kinematic.dvalue(hkin)[:, None, :],
            ),
        )
        hdiv[:, self.isotropic.nhist :, : self.isotropic.nhist] += torch.matmul(
            self.kinematic.dhistory_rate_derate(s, hkin, t, erate, T),
            torch.matmul(
                self.dflow_diso(s, h, t, T)[:, None, None],
                self.isotropic.dvalue(hiso)[:, None, :],
            ),
        )
        hdiv[:, self.isotropic.nhist :, self.isotropic.nhist :] += torch.matmul(
            self.kinematic.dhistory_rate_derate(s, hkin, t, erate, T),
            torch.matmul(
                self.dflow_dkin(s, h, t, T)[:, None, None],
                self.kinematic.dvalue(hkin)[:, None, :],
            ),
        )

        return hrate, hdiv

    def dflow_dhist(self, s, h, t, T):
        """
        The derivative of the flow rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        res = torch.zeros(s.shape[:1] + (1,) + h.shape[1:], device=h.device)

        hiso = h[:, : self.isotropic.nhist]
        hkin = h[:, self.isotropic.nhist :]

        res[:, 0, : self.isotropic.nhist] = self.dflow_diso(s, h, t, T)[
            :, None
        ] * self.isotropic.dvalue(hiso)
        res[:, 0, self.isotropic.nhist :] = self.dflow_dkin(s, h, t, T)[
            :, None
        ] * self.kinematic.dvalue(hkin)

        return res

    def dhist_dstress(self, s, h, t, T):
        """
        The derivative of the history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        res = torch.zeros(h.shape + s.shape[1:], device=h.device)

        erate, derate = self.flow_rate(s, h, t, T)

        hiso = h[:, : self.isotropic.nhist]
        hkin = h[:, self.isotropic.nhist :]

        res[:, : self.isotropic.nhist] = (
            self.isotropic.dhistory_rate_dstress(s, hiso, t, erate, T)
            + self.isotropic.dhistory_rate_derate(s, hiso, t, erate, T).bmm(
                derate[..., None, None]
            )[..., 0]
        )
        res[:, self.isotropic.nhist :] = (
            self.kinematic.dhistory_rate_dstress(s, hkin, t, erate, T)
            + self.kinematic.dhistory_rate_derate(s, hkin, t, erate, T).bmm(
                derate[..., None, None]
            )[..., 0]
        )

        return res
