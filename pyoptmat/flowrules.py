# pylint: disable=abstract-method, useless-super-delegation, line-too-long, duplicate-code, too-many-lines, too-many-arguments, too-many-public-methods

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

import numpy as np

import torch
from torch import nn

from pyoptmat import utility, solvers


class FlowRule(nn.Module):
    """
    Superclass for flow rule models

    This implementation provides default zero cross derivatives and that's it.
    """

    def __init__(self):
        super().__init__()

    def dflow_dhist(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the internal variables

        The superclass implementation provides a default of zero with the
        right shape.

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative of flow rate with respect to the
                              internal variables
        """
        return torch.zeros(s.shape + (1,) + h.shape[-1:], device=h.device)

    def dflow_derate(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the total strain rate

        The superclass implementation provides a default of zero with the
        right shape.

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative of flow rate with respect to the
                              internal variables
        """
        return torch.zeros(s.shape, device=h.device)

    def dhist_dstress(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the stress

        The superclass implementation provides a default of zero with the
        right shape.

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative of flow rate with respect to the stress
        """
        return torch.zeros(h.shape, device=h.device)

    def dhist_derate(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the total strain rate

        The superclass implementation provides a default of zero with the
        right shape.

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative of flow rate with respect to the strain rate
        """
        return torch.zeros(h.shape, device=h.device)


class KocksMeckingRegimeFlowRule(FlowRule):
    """
    Switches between two different flow rules depending on the value of the
    Kocks-Mecking normalized activation energy

    .. math::

        g = \\frac{kT}{\\mu b^3} \\log{\\frac{\\dot{\\varepsilon}_0}{\\dot{\\varepsilon}}}

    with :math:`k` the Boltzmann constant, :math:`T` temperature, :math:`\\mu` the shear modulus,
    :math:`b` a representative Burgers vector, :math:`\\dot{\\varepsilon}_0` a reference
    strain rate, and :math:`\\dot{\\varepsilon}` the current applied strain rate.

    If the activation energy is less than or equal to a threshold :math:`g_0` the flow rate
    is equal to that of the first model.  If the activation energy is greater than the
    threshold then the model switches to the second flow rule.

    The two models should have compatible internal history vectors.

    Args:
        model1 (flowrules.FlowRule):            first flow rule
        model2 (flowrules.FlowRule):            second flow rule
        g0 (torch.tensor):                      activation energy threshold
        mu (temperature.TemperatureParameter):  shear modulus
        b (torch.tensor):                       burgers vector
        eps0 (torch.tensor):                    reference strain rate
        k (torch.tensor):                       Boltzmann constant

    Keyword Args:
        eps (float):                            default 1e-20, offset to
                                                avoid divide-by-zero
        g0_scale (function):                    scaling function for g0,
                                                defaults to no scaling
    """

    def __init__(
        self,
        model1,
        model2,
        g0,
        mu,
        b,
        eps0,
        k,
        eps=torch.tensor(1e-20),
        g0_scale=lambda x: x,
    ):
        super().__init__()

        self.model1 = model1
        self.model2 = model2
        self.g0 = g0

        self.mu = mu
        self.b = b
        self.eps0 = eps0
        self.k = k

        self.eps = eps
        self.g0_scale = g0_scale

        # Check for conformal history vectors
        if self.model1.nhist != self.model2.nhist:
            raise ValueError(
                "The two models provided to the KocksMeckingRegimeFlowRule "
                "model must have the same number of internal variables"
            )

    @property
    def nhist(self):
        """
        The number of internal variables
        """
        return self.model1.nhist

    def g(self, T, e):
        """
        The current value of activation energy

        Args:
            T (torch.tensor):   temperature
            e (torch.tensor):   total strain rate

        Returns:
            torch.tensor:       value of the activation energy
        """
        return (
            self.k
            * T
            / (self.mu.value(T) * self.b**3.0)
            * torch.log(self.eps0 / (torch.abs(e) + self.eps))
        )

    def switch_values(self, vals1, vals2, T, e):
        """
        Switch between the two model results

        Args:
            vals1 (torch.tensor):   values from first model
            vals2 (torch.tensor):   values from second model
            T (torch.tensor):       temperatures
            e (torch.tensor):       strain rates

        """
        result = torch.clone(vals1)
        second = self.g(T, e) > self.g0_scale(self.g0)
        result[second] = vals2[second]

        return result

    def flow_rate(self, s, h, t, T, e):
        """
        The uniaxial flow rate itself and the derivative
        with respect to stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):    the flow rate and the derivative
                                                of the flow rate with
                                                respect to stress
        """
        flow1, dflow1 = self.model1.flow_rate(s, h, t, T, e)
        flow2, dflow2 = self.model2.flow_rate(s, h, t, T, e)

        return (
            self.switch_values(flow1, flow2, T, e),
            self.switch_values(dflow1, dflow2, T, e),
        )

    def dflow_dhist(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        return self.switch_values(
            self.model1.dflow_dhist(s, h, t, T, e),
            self.model2.dflow_dhist(s, h, t, T, e),
            T,
            e,
        )

    def dflow_derate(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the total strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative of flow rate with respect to the
                              internal variables
        """
        return self.switch_values(
            self.model1.dflow_derate(s, h, t, T, e),
            self.model2.dflow_derate(s, h, t, T, e),
            T,
            e,
        )

    def history_rate(self, s, h, t, T, e):
        """
        The history rate and the derivative of the history rate with respect
        to the current history

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):    the history rate and the
                                                derivative of the history rate
                                                with respect to history
        """
        rate1, drate1 = self.model1.history_rate(s, h, t, T, e)
        rate2, drate2 = self.model2.history_rate(s, h, t, T, e)

        return (
            self.switch_values(rate1, rate2, T, e),
            self.switch_values(drate1, drate2, T, e),
        )

    def dhist_dstress(self, s, h, t, T, e):
        """
        The derivative of the history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        return self.switch_values(
            self.model1.dhist_dstress(s, h, t, T, e),
            self.model2.dhist_dstress(s, h, t, T, e),
            T,
            e,
        )

    def dhist_derate(self, s, h, t, T, e):
        """
        The derivative of the history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        return self.switch_values(
            self.model1.dhist_derate(s, h, t, T, e),
            self.model2.dhist_derate(s, h, t, T, e),
            T,
            e,
        )


class SoftKocksMeckingRegimeFlowRule(FlowRule):
    """
    Switches between two different flow rules depending on the value of the
    Kocks-Mecking normalized activation energy

    .. math::

        g = \\frac{kT}{\\mu b^3} \\log{\\frac{\\dot{\\varepsilon}_0}{\\dot{\\varepsilon}}}

    with :math:`k` the Boltzmann constant, :math:`T` temperature, :math:`\\mu` the shear modulus,
    :math:`b` a representative Burgers vector, :math:`\\dot{\\varepsilon}_0` a reference
    strain rate, and :math:`\\dot{\\varepsilon}` the current applied strain rate.

    If the activation energy is less than or equal to a threshold :math:`g_0` the flow rate
    is equal to that of the first model.  If the activation energy is greater than the
    threshold then the model switches to the second flow rule.

    The two models should have compatible internal history vectors.

    This version uses a soft blending of the two models rather than a hard switch.  Specifically,

    .. math::

        M = f M_1 + (1-f) M_2

    with

    .. math::

        f = \\frac{\\tanh \\left[s_f \\left(g - g_0 \\right) \\right] + 1}{2}

    where :math:`s_f` is some scaling function, with values around 100 providing
    a fairly hard switch between the two models

    Args:
        model1 (flowrules.FlowRule):            first flow rule
        model2 (flowrules.FlowRule):            second flow rule
        A, B, C (torch.tensor):                 Kocks-Mecking parameters
        mu (temperature.TemperatureParameter):  shear modulus
        b (torch.tensor):                       burgers vector
        eps0 (torch.tensor):                    reference strain rate
        k (torch.tensor):                       Boltzmann constant
        sf (torch.tensor):                      sharpness parameter

    Keyword Args:
        A_scale, B_scale, C_scale:              scaling functions for K-M parameters
        eps (float):                            default 1e-20, offset to
                                                avoid divide-by-zero
    """

    def __init__(
        self,
        model1,
        model2,
        A,
        B,
        C,
        mu,
        b,
        eps0,
        k,
        sf,
        A_scale=lambda x: x,
        B_scale=lambda x: x,
        C_scale=lambda x: x,
        eps=torch.tensor(1e-20),
    ):
        super().__init__()

        self.model1 = model1
        self.model2 = model2
        self.A = A
        self.B = B
        self.C = C

        self.mu = mu
        self.b = b
        self.eps0 = eps0
        self.k = k

        self.sf = sf

        self.eps = eps
        self.A_scale = A_scale
        self.B_scale = B_scale
        self.C_scale = C_scale

        # Check for conformal history vectors
        if self.model1.nhist != self.model2.nhist:
            raise ValueError(
                "The two models provided to the KocksMeckingRegimeFlowRule "
                "model must have the same number of internal variables"
            )

    @property
    def nhist(self):
        """
        The number of internal variables
        """
        return self.model1.nhist

    def g(self, T, e):
        """
        The current value of activation energy

        Args:
            T (torch.tensor):   temperature
            e (torch.tensor):   total strain rate

        Returns:
            torch.tensor:       value of the activation energy
        """
        return (
            self.k
            * T
            / (self.mu.value(T) * self.b**3.0)
            * torch.log(self.eps0 / (torch.abs(e) + self.eps))
        )

    def dg_e(self, T, e):
        """
        Derivative of the activation energy with respect to
        the strain rate

        Args:
            T (torch.tensor):   temperature
            e (torch.tensor):   total strain rate

        Returns:
            torch.tensor:       derivative of the activation energy
        """
        return -(
            self.k * T / (self.mu.value(T) * self.b**3.0) / (torch.abs(e) + self.eps)
        ) * torch.sign(e)

    def f(self, T, e):
        """
        The weight function value

        Args:
            T (torch.tensor):   temperature
            e (torch.tensor):   total strain rate

        Returns:
            torch.tensor:       value of the weighting function
        """
        return (torch.tanh(self.sf * (self.g(T, e) - self.g0())) + 1.0) / 2.0

    def g0(self):
        """
        The intercept value
        """
        return (self.C_scale.scale(self.C) - self.B_scale.scale(self.B)) / self.A_scale(
            self.A
        )

    def df_e(self, T, e):
        """
        The derivative of the weight function with respect to
        the strain rate

        Args:
            T (torch.tensor):   temperature
            e (torch.tensor):   total strain rate

        Returns:
            torch.tensor:       derivative of the weight function
        """
        return (
            self.sf
            / (2.0 * torch.cosh(self.sf * (self.g(T, e) - self.g0())) ** 2.0)
            * self.dg_e(T, e)
        )

    def blend_values(self, vals1, vals2, T, e):
        """
        Switch between the two model results

        Args:
            vals1 (torch.tensor):   values from first model
            vals2 (torch.tensor):   values from second model
            T (torch.tensor):       temperatures
            e (torch.tensor):       strain rates
        """
        f = self.f(T, e)
        # Ugh, really?
        diff = vals1.dim() - f.dim()
        f = f[(...,) + (None,) * diff]

        return (1 - f) * vals1 + f * vals2

    def flow_rate(self, s, h, t, T, e):
        """
        The uniaxial flow rate itself and the derivative
        with respect to stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):    the flow rate and the derivative
                                                of the flow rate with
                                                respect to stress
        """
        flow1, dflow1 = self.model1.flow_rate(s, h, t, T, e)
        flow2, dflow2 = self.model2.flow_rate(s, h, t, T, e)

        return (
            self.blend_values(flow1, flow2, T, e),
            self.blend_values(dflow1, dflow2, T, e),
        )

    def dflow_dhist(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        return self.blend_values(
            self.model1.dflow_dhist(s, h, t, T, e),
            self.model2.dflow_dhist(s, h, t, T, e),
            T,
            e,
        )

    def dflow_derate(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the total strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative of flow rate with respect to the
                              internal variables
        """
        flow1, _ = self.model1.flow_rate(s, h, t, T, e)
        flow2, _ = self.model2.flow_rate(s, h, t, T, e)
        df = self.df_e(T, e)

        return self.blend_values(
            self.model1.dflow_derate(s, h, t, T, e),
            self.model2.dflow_derate(s, h, t, T, e),
            T,
            e,
        ) + df * (flow2 - flow1)

    def history_rate(self, s, h, t, T, e):
        """
        The history rate and the derivative of the history rate with respect
        to the current history

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):    the history rate and the
                                                derivative of the history rate
                                                with respect to history
        """
        rate1, drate1 = self.model1.history_rate(s, h, t, T, e)
        rate2, drate2 = self.model2.history_rate(s, h, t, T, e)

        return (
            self.blend_values(rate1, rate2, T, e),
            self.blend_values(drate1, drate2, T, e),
        )

    def dhist_dstress(self, s, h, t, T, e):
        """
        The derivative of the history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        return self.blend_values(
            self.model1.dhist_dstress(s, h, t, T, e),
            self.model2.dhist_dstress(s, h, t, T, e),
            T,
            e,
        )

    def dhist_derate(self, s, h, t, T, e):
        """
        The derivative of the history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        rate1, _ = self.model1.history_rate(s, h, t, T, e)
        rate2, _ = self.model2.history_rate(s, h, t, T, e)

        df = self.df_e(T, e).unsqueeze(-1)

        return (
            self.blend_values(
                self.model1.dhist_derate(s, h, t, T, e),
                self.model2.dhist_derate(s, h, t, T, e),
                T,
                e,
            )
            + (rate2 - rate1) * df
        )


class RateIndependentFlowRuleWrapper(FlowRule):
    """
    Wraps another flow rule using Walker's time dilation trick to make it
    behave as if it was rate-independent.

    Specifically, the model dilates time with the formula:

    .. math::

        \\Delta t \\rightarrow \\kappa \\Delta t

    with

    .. math::

        \\kappa = 1 - \\lambda + \\frac{\\lambda \\left| \\dot{\\varepsilon} \\right|}{\\dot{\\varepsilon}_{ref}}

    where :math:`\\lambda` is a parameter where :math:`\\lambda = 0` gives
    the original, rate dependent response and :math:`\\lambda \\approx 1`
    gives an approximately rate independent response,
    :math:`\\dot{\\varepsilon}_{ref}` is a reference strain rate which
    should be several orders of magnitude smaller than the applied strain rate,
    and :math:`\\dot{\\varepsilon}` is the current, transient
    strain rate applied to the model.

    Args:
        base (flowrules.FlowRule):  the base model
        lmbda (scalar):             the tuning parameter :math:`\\lambda`
        eps_ref (scalar):           the reference strain rate :math:`\\dot{\\varepsilon}_{ref}`

    """

    def __init__(self, base, lmbda, eps_ref):
        super().__init__()

        self.base = base
        self.lmbda = lmbda
        self.eps_ref = eps_ref

    @property
    def nhist(self):
        """
        The number of internal variables
        """
        return self.base.nhist

    def scale(self, e):
        """
        The current value of the time dilation factor

        Args:
            e (torch.tensor):   total strain rate

        Returns:
            torch.tensor:       current scale factor
        """
        return 1.0 - self.lmbda + self.lmbda * torch.abs(e) / self.eps_ref

    def dscale(self, e):
        """
        The derivative of the current value of the time
        dilation factor with respect to the total strain
        rate

        Args:
            e (torch.tensor):   total strain rate

        Returns:
            torch.tensor:       derivative of the scale factor
                                with respect to the total strain rate
        """
        return torch.sign(e) * self.lmbda / self.eps_ref

    def flow_rate(self, s, h, t, T, e):
        """
        The uniaxial flow rate itself and the derivative
        with respect to stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):    the flow rate and the derivative
                                                of the flow rate with
                                                respect to stress
        """
        base, dbase = self.base.flow_rate(s, h, t, T, e)
        sf = self.scale(e)

        return sf * base, sf * dbase

    def dflow_dhist(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        return self.base.dflow_dhist(s, h, t, T, e) * self.scale(e)[..., None, None]

    def dflow_derate(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the total strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative of flow rate with respect to the
                              internal variables
        """
        return self.base.dflow_derate(s, h, t, T, e) * self.scale(
            e
        ) + self.base.flow_rate(s, h, t, T, e)[0] * self.dscale(e)

    def history_rate(self, s, h, t, T, e):
        """
        The history rate and the derivative of the history rate with respect
        to the current history

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):    the history rate and the
                                                derivative of the history rate
                                                with respect to history
        """
        rate, drate = self.base.history_rate(s, h, t, T, e)
        sf = self.scale(e)

        return sf[..., None] * rate, sf[..., None, None] * drate

    def dhist_dstress(self, s, h, t, T, e):
        """
        The derivative of the history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        return self.scale(e)[..., None] * self.base.dhist_dstress(s, h, t, T, e)

    def dhist_derate(self, s, h, t, T, e):
        """
        The derivative of the history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        return (
            self.base.dhist_derate(s, h, t, T, e) * self.scale(e)[..., None]
            + self.base.history_rate(s, h, t, T, e)[0] * self.dscale(e)[..., None]
        )


class SuperimposedFlowRule(FlowRule):
    """
    Superimpose multiple flow rules with

    .. math::

        \\dot{\\varepsilon}_{in}=\\sum_i \\dot{\\varepsilon}_{in,i}

    and the history the union of all the individual models

    Args:
        models (list):      list of the individual models

    """

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

        self.nmodels = len(self.models)
        self.nhist_per = [m.nhist for m in self.models]
        self.offsets = [0] + list(np.cumsum(self.nhist_per))[:-1]

    @property
    def nhist(self):
        """
        The number of internal variables
        """
        return sum(self.nhist_per)

    def flow_rate(self, s, h, t, T, e):
        """
        The uniaxial flow rate itself and the derivative
        with respect to stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):    the flow rate and the derivative
                                                of the flow rate with
                                                respect to stress
        """
        rate = torch.zeros_like(s)
        drate = torch.zeros_like(s)

        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            ratei, dratei = model.flow_rate(s, h[..., o : o + n], t, T, e)
            rate += ratei
            drate += dratei

        return (rate, drate)

    def dflow_dhist(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        res = torch.zeros(s.shape + (1,) + h.shape[-1:], device=h.device)

        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            dratei = model.dflow_dhist(s, h[..., o : o + n], t, T, e)
            res[..., :, o : o + n] = dratei

        return res

    def dflow_derate(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the total strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative of flow rate with respect to the
                              internal variables
        """
        res = torch.zeros(e.shape, device=h.device)

        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            dratei = model.dflow_derate(s, h[..., o : o + n], t, T, e)
            res += dratei

        return res

    def history_rate(self, s, h, t, T, e):
        """
        The history rate and the derivative of the history rate with respect
        to the current history

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):    the history rate and the
                                                derivative of the history rate
                                                with respect to history
        """
        rate = torch.zeros_like(h)
        drate = torch.zeros(h.shape + h.shape[-1:], device=h.device)

        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            ratei, dratei = model.history_rate(s, h[..., o : o + n], t, T, e)
            rate[..., o : o + n] = ratei
            drate[..., o : o + n, o : o + n] = dratei

        return (rate, drate)

    def dhist_dstress(self, s, h, t, T, e):
        """
        The derivative of the history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        res = torch.zeros(h.shape, device=h.device)

        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            dratei = model.dhist_dstress(s, h[..., o : o + n], t, T, e)
            res[..., o : o + n] = dratei

        return res

    def dhist_derate(self, s, h, t, T, e):
        """
        The derivative of the history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        res = torch.zeros(h.shape, device=h.device)

        for o, n, model in zip(self.offsets, self.nhist_per, self.models):
            dratei = model.dhist_derate(s, h[..., o : o + n], t, T, e)
            res[..., o : o + n] = dratei

        return res


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

    def flow_rate(self, s, h, t, T, e):
        """
        The uniaxial flow rate itself and the derivative
        with respect to stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

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

    def history_rate(self, s, h, t, T, e):
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


class PerfectRateIndependentPlasticity(FlowRule):
    """
    Perfect rate independent plasticity

    Args:
      sy (|TP|):     yield stress
    """

    def __init__(self, sy):
        super().__init__()
        self.sy = sy

    def f(self, s, h, T):
        """
        The yield function

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          T (torch.tensor):   temperature

        Returns:
          the value of the yield function...
        """
        return torch.abs(s) - self.sy(T)

    def flow_rate(self, s, h, t, T, e):
        """
        The uniaxial flow rate itself and the derivative
        with respect to stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):    the flow rate and the derivative
                                                of the flow rate with
                                                respect to stress
        """
        fr = torch.zeros_like(s)
        yielding = self.f(s, h, T) > 0
        fr[yielding] = e[yielding]

        dfr = torch.zeros_like(s)

        return fr, dfr

    def dflow_derate(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the total strain rate

        The superclass implementation provides a default of zero with the
        right shape.

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative of flow rate with respect to the
                              internal variables
        """
        fr = torch.zeros_like(s)
        yielding = self.f(s, h, T) > 0
        fr[yielding] = 1.0

        return fr

    @property
    def nhist(self):
        """
        The number of internal variables

        Here 0...
        """
        return 0

    def history_rate(self, s, h, t, T, e):
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

    def flow_rate(self, s, h, t, T, e):
        """
        The flow rate itself and the derivative with respect to stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):  the flow rate and the derivative of the flow rate with
                                              respect to stress
        """
        ih = self.isotropic.value(h[..., : self.isotropic.nhist])
        kh = self.kinematic.value(h[..., self.isotropic.nhist :])

        return (
            utility.macaulay((torch.abs(s - kh) - self.s0(T) - ih) / self.eta(T))
            ** self.n(T)
            * torch.sign(s - kh),
            self.n(T)
            * utility.macaulay((torch.abs(s - kh) - self.s0(T) - ih) / self.eta(T))
            ** (self.n(T) - 1)
            / self.eta(T),
        )

    def dflow_diso(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the isotropic hardening

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        ih = self.isotropic.value(h[..., : self.isotropic.nhist])
        kh = self.kinematic.value(
            h[..., self.isotropic.nhist : self.isotropic.nhist + self.kinematic.nhist]
        )

        iv = (torch.abs(s - kh) - self.s0(T) - ih) / self.eta(T)

        return (
            -self.n(T)
            * utility.macaulay(iv) ** (self.n(T) - 1)
            / self.eta(T)
            * torch.sign(s - kh)
        )

    def dflow_dkin(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the kinematic hardening

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        ih = self.isotropic.value(h[..., : self.isotropic.nhist])
        kh = self.kinematic.value(
            h[..., self.isotropic.nhist : self.isotropic.nhist + self.kinematic.nhist]
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

    def history_rate(self, s, h, t, T, e):
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
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):    the history rate and the
                                                derivative of the history rate
                                                with respect to history
        """
        erate, _ = self.flow_rate(s, h, t, T, e)

        hiso = h[..., : self.isotropic.nhist]
        hkin = h[..., self.isotropic.nhist :]

        # History evolution rate
        hrate = torch.cat(
            [
                self.isotropic.history_rate(s, hiso, t, erate, T, e),
                self.kinematic.history_rate(s, hkin, t, erate, T, e),
            ],
            dim=-1,
        )

        # Jacobian contribution
        hdiv = torch.cat(
            [
                torch.cat(
                    [
                        self.isotropic.dhistory_rate_dhistory(s, hiso, t, erate, T, e)
                        + utility.mbmm(
                            self.isotropic.dhistory_rate_derate(
                                s, hiso, t, erate, T, e
                            ),
                            utility.mbmm(
                                self.dflow_diso(s, h, t, T, e)[..., None, None],
                                self.isotropic.dvalue(hiso)[..., None, :],
                            ),
                        ),
                        self.isotropic.dhistory_rate_derate(s, hiso, t, erate, T, e)
                        * self.dflow_dkin(s, h, t, T, e)[..., None, None]
                        * self.kinematic.dvalue(hkin)[..., None, :],
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [
                        utility.mbmm(
                            self.kinematic.dhistory_rate_derate(
                                s, hkin, t, erate, T, e
                            ),
                            utility.mbmm(
                                self.dflow_diso(s, h, t, T, e)[..., None, None],
                                self.isotropic.dvalue(hiso)[..., None, :],
                            ),
                        ),
                        self.kinematic.dhistory_rate_dhistory(s, hkin, t, erate, T, e)
                        + utility.mbmm(
                            self.kinematic.dhistory_rate_derate(
                                s, hkin, t, erate, T, e
                            ),
                            utility.mbmm(
                                self.dflow_dkin(s, h, t, T, e)[..., None, None],
                                self.kinematic.dvalue(hkin)[..., None, :],
                            ),
                        ),
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )

        return hrate, hdiv

    def dflow_dhist(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        hiso = h[..., : self.isotropic.nhist]
        hkin = h[..., self.isotropic.nhist :]

        return torch.cat(
            [
                self.dflow_diso(s, h, t, T, e)[..., None] * self.isotropic.dvalue(hiso),
                self.dflow_dkin(s, h, t, T, e)[..., None] * self.kinematic.dvalue(hkin),
            ],
            dim=-1,
        ).unsqueeze(-2)

    def dhist_dstress(self, s, h, t, T, e):
        """
        The derivative of the history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        erate, derate = self.flow_rate(s, h, t, T, e)

        hiso = h[..., : self.isotropic.nhist]
        hkin = h[..., self.isotropic.nhist :]

        return torch.cat(
            [
                self.isotropic.dhistory_rate_dstress(s, hiso, t, erate, T, e)
                + utility.mbmm(
                    self.isotropic.dhistory_rate_derate(s, hiso, t, erate, T, e),
                    derate[..., None, None],
                )[..., 0],
                self.kinematic.dhistory_rate_dstress(s, hkin, t, erate, T, e)
                + utility.mbmm(
                    self.kinematic.dhistory_rate_derate(s, hkin, t, erate, T, e),
                    derate[..., None, None],
                )[..., 0],
            ],
            dim=-1,
        )

    def dhist_derate(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the total strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative of flow rate with respect to the strain rate
        """
        erate, _ = self.flow_rate(s, h, t, T, e)

        hiso = h[..., : self.isotropic.nhist]
        hkin = h[..., self.isotropic.nhist :]

        return torch.cat(
            [
                self.isotropic.dhistory_rate_dtotalrate(s, hiso, t, erate, T, e),
                self.kinematic.dhistory_rate_dtotalrate(s, hkin, t, erate, T, e),
            ],
            dim=-1,
        )


class IsoKinRateIndependentPlasticity(FlowRule):
    """
    An approximation to rate independent plasticity.  The model is defined by

    .. math::

        \\dot{\\varepsilon}_{in} = \\xi(f) \\dot{\\varepsilon}_{p,ri}

    where :math:`\\xi` is a sigmoid function, :math:`f` is a flow surface
    and :math:`\\dot{\\varepsilon}_{p,ri}` is the rate independent plastic flow
    rate, as defined by the classical consistency conditions.

    This function uses the flow surface

    .. math::

        f = \\left|\\sigma - x\\right| - \\sigma_y - k

    and where the :py:class:`pyoptmat.hardening.IsotropicHardeningModel` and
    :py:class:`pyoptmat.hardening.KinematicHardeningModel` objects determine the
    history rate of isotropic (:math:`k`) and kinematic (:math:`x`) hardening.

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
      E (|TP|):         young's modulus, needed to define the rate independent flow rate
      sy (|TP|):        yield stress
      isotropic (:py:class:`pyoptmat.hardening.IsotropicHardeningModel`): object providing the isotropic hardening model
      kinematic (:py:class:`pyoptmat.hardening.IsotropicHardeningModel`): object providing the kinematic hardening model

    Keyword Args:
        soffset (float):    small offset to the stress in the  yield surface to avoid a singularity at zero
        s (float):          scale factor for the sigmoid function, controls the amount of smoothing at the onset of plasticity
    """

    def __init__(self, E, sy, isotropic, kinematic, soffset=1e-10, s=1.0):
        super().__init__()
        self.E = E
        self.isotropic = isotropic
        self.kinematic = kinematic
        self.sy = sy
        self.soffset = soffset

        self.s = s

    def f(self, s, h, T):
        """
        The yield function

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          T (torch.tensor):   temperature

        Returns:
          the value of the yield function
        """
        ih = self.isotropic.value(h[..., : self.isotropic.nhist])
        kh = self.kinematic.value(h[..., self.isotropic.nhist :])
        return torch.abs(s - kh) - self.sy(T) - ih

    def df_ds(self, s, h, T):
        """
        The derivative of the yield function with respect to stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          T (torch.tensor):   temperature

        Returns:
          the derivative value
        """
        kh = self.kinematic.value(h[..., self.isotropic.nhist :])
        return torch.sign(s - kh)

    def df_dh(self, s, h, T):
        """
        The derivative of the yield function with respect to the history

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          T (torch.tensor):   temperature

        Returns:
          the derivative value
        """
        kh = self.kinematic.value(h[..., self.isotropic.nhist :])
        di = -self.isotropic.dvalue(h[..., : self.isotropic.nhist])
        dk = -torch.sign(s - kh).unsqueeze(-1) * self.kinematic.dvalue(
            h[..., self.isotropic.nhist :]
        )

        return torch.cat([di, dk], axis=-1)

    def ep_residual(self, ep, s, h, t, T, e):
        """
        The residual function to solve for the consistency parameter, here just
        expanded to the plastic flow rate as it's a scalar.

        Args:
          ep (torch.tensor):  current values of the plastic flow rate
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   current time
          T (torch.tensor):   temperature
          e (torch.tensor):   current total strain rate

        Returns:
          the residual value
        """
        hiso = h[..., : self.isotropic.nhist]
        hkin = h[..., self.isotropic.nhist :]

        kh = self.kinematic.value(hkin)

        i_dot = (
            utility.mbmm(
                self.isotropic.dvalue(hiso).unsqueeze(-2),
                self.isotropic.history_rate(s, hiso, t, ep, T, e).unsqueeze(-1),
            )
            .squeeze(-1)
            .squeeze(-1)
        )
        k_dot = (
            utility.mbmm(
                self.kinematic.dvalue(hkin).unsqueeze(-2),
                self.kinematic.history_rate(s, hkin, t, ep, T, e).unsqueeze(-1),
            )
            .squeeze(-1)
            .squeeze(-1)
        )

        return (
            torch.sign(s - kh + self.soffset) * self.E(T) * (e - ep)
            - torch.sign(s - kh + self.soffset) * k_dot
            - i_dot
        )

    def ep_jacobian_ep(self, ep, s, h, t, T, e):
        """
        The Jacobian of the consistency residual with respect to the plastic strain rate.

        Args:
          ep (torch.tensor):  current values of the plastic flow rate
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   current time
          T (torch.tensor):   temperature
          e (torch.tensor):   current total strain rate

        Returns:
          the derivative value
        """
        hiso = h[..., : self.isotropic.nhist]
        hkin = h[..., self.isotropic.nhist :]

        kh = self.kinematic.value(hkin)

        di_dot = utility.mbmm(
            self.isotropic.dvalue(hiso).unsqueeze(-2),
            self.isotropic.dhistory_rate_derate(s, hiso, t, ep, T, e),
        ).squeeze(-1)
        dk_dot = utility.mbmm(
            self.kinematic.dvalue(hkin).unsqueeze(-2),
            self.kinematic.dhistory_rate_derate(s, hkin, t, ep, T, e),
        ).squeeze(-1)

        return (
            -(torch.sign(s - kh + self.soffset) * self.E(T)).unsqueeze(-1)
            - torch.sign(s - kh + self.soffset).unsqueeze(-1) * dk_dot
            - di_dot
        )

    def ep_jacobian_s(self, ep, s, h, t, T, e):
        """
        The Jacobian of the consistency residual with respect to the stress

        Args:
          ep (torch.tensor):  current values of the plastic flow rate
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   current time
          T (torch.tensor):   temperature
          e (torch.tensor):   current total strain rate

        Returns:
          the derivative value
        """
        hiso = h[..., : self.isotropic.nhist]
        hkin = h[..., self.isotropic.nhist :]

        kh = self.kinematic.value(hkin)

        di_dot = utility.mbmm(
            self.isotropic.dvalue(hiso).unsqueeze(-2),
            self.isotropic.dhistory_rate_dstress(s, hiso, t, ep, T, e).unsqueeze(-1),
        ).squeeze(-1)
        dk_dot = utility.mbmm(
            self.kinematic.dvalue(hkin).unsqueeze(-2),
            self.kinematic.dhistory_rate_dstress(s, hkin, t, ep, T, e).unsqueeze(-1),
        ).squeeze(-1)

        return -torch.sign(s - kh + self.soffset).unsqueeze(-1) * dk_dot - di_dot

    def ep_jacobian_h(self, ep, s, h, t, T, e):
        """
        The Jacobian of the consistency residual with respect to the history

        Args:
          ep (torch.tensor):  current values of the plastic flow rate
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   current time
          T (torch.tensor):   temperature
          e (torch.tensor):   current total strain rate

        Returns:
          the derivative value
        """
        hiso = h[..., : self.isotropic.nhist]
        hkin = h[..., self.isotropic.nhist :]

        kh = self.kinematic.value(hkin)

        di_dot = utility.mbmm(
            self.isotropic.dvalue(hiso).unsqueeze(-2),
            self.isotropic.dhistory_rate_dhistory(s, hiso, t, ep, T, e),
        ).squeeze(-2)
        dk_dot = utility.mbmm(
            self.kinematic.dvalue(hkin).unsqueeze(-2),
            self.kinematic.dhistory_rate_dhistory(s, hkin, t, ep, T, e),
        ).squeeze(-2)

        return torch.cat([-di_dot, -torch.sign(s - kh).unsqueeze(-1) * dk_dot], dim=-1)

    def ep_jacobian_e(self, ep, s, h, t, T, e):
        """
        The Jacobian of the consistency residual with respect to the total strain rate

        Args:
          ep (torch.tensor):  current values of the plastic flow rate
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   current time
          T (torch.tensor):   temperature
          e (torch.tensor):   current total strain rate

        Returns:
          the derivative value
        """
        hkin = h[..., self.isotropic.nhist :]

        kh = self.kinematic.value(hkin)

        return (torch.sign(s - kh + self.soffset) * self.E(T)).unsqueeze(-1)

    def sig(self, x):
        """
        Our chosen sigmoid function

        .. math::

            \\xi = \\frac{\\tanh(s x) + 1}{2}

        with :math:`s` the scale factor

        Args:
            x (torch.tensor): input to sigmoid
        """
        return (torch.tanh(self.s * x) + 1.0) / 2.0

    def dsig(self, x):
        """
        Derivative of the sigmoid

        Args:
            x (torch.tensor): input to sigmoid
        """
        return 0.5 * self.s / torch.cosh(self.s * x) ** 2.0

    def mix_fn(self, s, h, T):
        """
        The mixture function :math:`\\xi(f)`

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          T (torch.tensor):   temperature
        """
        return self.sig(self.f(s, h, T))

    def dmix_fn_ds(self, s, h, T):
        """
        The derivative of the mixture function with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          T (torch.tensor):   temperature
        """
        return self.dsig(self.f(s, h, T)) * self.df_ds(s, h, T)

    def dmix_fn_dh(self, s, h, T):
        """
        The derivative of the mixture function with respect to the history

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          T (torch.tensor):   temperature
        """
        return self.dsig(self.f(s, h, T)).unsqueeze(-1) * self.df_dh(s, h, T)

    def plastic_rate(self, s, h, t, T, e):
        """
        Solve for the plastic strain rate that meets the consistency criteria

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate
        """

        def RJ(x):
            return self.ep_residual(
                x, s + self.soffset, h, t, T, e
            ), self.ep_jacobian_ep(x, s + self.soffset, h, t, T, e).squeeze(-1)

        return solvers.scalar_newton(RJ, torch.zeros_like(e))

    def flow_rate(self, s, h, t, T, e):
        """
        The flow rate itself and the derivative with respect to stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):  the flow rate and the derivative of the flow rate with
                                              respect to stress
        """
        # Solve for the plastic flow rate
        mf = self.mix_fn(s, h, T)
        bfr = self.plastic_rate(s, h, t, T, e)

        fr = mf * bfr
        dfr = (
            -(
                self.ep_jacobian_s(bfr, s, h, t, T, e)
                / self.ep_jacobian_ep(bfr, s, h, t, T, e)
            ).squeeze(-1)
            * mf
            + self.dmix_fn_ds(s, h, T) * bfr
        )

        return fr, dfr

    def dflow_derate(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the total strain rate

        The superclass implementation provides a default of zero with the
        right shape.

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative of flow rate with respect to the
                              internal variables
        """
        pfr = self.plastic_rate(s, h, t, T, e)
        mf = self.mix_fn(s, h, T)
        dfr = (
            -(
                self.ep_jacobian_e(pfr, s, h, t, T, e)
                / self.ep_jacobian_ep(pfr, s, h, t, T, e)
            ).squeeze(-1)
            * mf
        )

        return dfr

    def dflow_dhist(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        bfr = self.plastic_rate(s, h, t, T, e)
        mf = self.mix_fn(s, h, T)
        dfr = -(
            self.ep_jacobian_h(bfr, s, h, t, T, e)
            / self.ep_jacobian_ep(bfr, s, h, t, T, e)
        ) * mf.unsqueeze(-1) + self.dmix_fn_dh(s, h, T) * bfr.unsqueeze(-1)

        return dfr.unsqueeze(-2)

    @property
    def nhist(self):
        """
        The number of internal variables, here the sum from the isotropic
        and kinematic hardening models
        """
        return self.isotropic.nhist + self.kinematic.nhist

    def history_rate(self, s, h, t, T, e):
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
          e (torch.tensor):   total strain rate

        Returns:
          tuple(torch.tensor, torch.tensor):    the history rate and the
                                                derivative of the history rate
                                                with respect to history
        """
        erate, _ = self.flow_rate(s, h, t, T, e)

        hiso = h[..., : self.isotropic.nhist]
        hkin = h[..., self.isotropic.nhist :]

        # History evolution rate
        hrate = torch.cat(
            [
                self.isotropic.history_rate(s, hiso, t, erate, T, e),
                self.kinematic.history_rate(s, hkin, t, erate, T, e),
            ],
            dim=-1,
        )

        df_dh = self.dflow_dhist(s, h, t, T, e)
        df_di = df_dh[..., : self.isotropic.nhist]
        df_dk = df_dh[..., self.isotropic.nhist :]

        # Jacobian contribution
        hdiv = torch.cat(
            [
                torch.cat(
                    [
                        self.isotropic.dhistory_rate_dhistory(s, hiso, t, erate, T, e)
                        + utility.mbmm(
                            self.isotropic.dhistory_rate_derate(
                                s, hiso, t, erate, T, e
                            ),
                            df_di,
                        ),
                        self.isotropic.dhistory_rate_derate(s, hiso, t, erate, T, e)
                        * df_dk,
                    ],
                    dim=-1,
                ),
                torch.cat(
                    [
                        utility.mbmm(
                            self.kinematic.dhistory_rate_derate(
                                s, hkin, t, erate, T, e
                            ),
                            df_di,
                        ),
                        self.kinematic.dhistory_rate_dhistory(s, hkin, t, erate, T, e)
                        + utility.mbmm(
                            self.kinematic.dhistory_rate_derate(
                                s, hkin, t, erate, T, e
                            ),
                            df_dk,
                        ),
                    ],
                    dim=-1,
                ),
            ],
            dim=-2,
        )

        return hrate, hdiv

    def dhist_dstress(self, s, h, t, T, e):
        """
        The derivative of the history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       the derivative of the flow rate
        """
        erate, derate = self.flow_rate(s, h, t, T, e)

        hiso = h[..., : self.isotropic.nhist]
        hkin = h[..., self.isotropic.nhist :]

        return torch.cat(
            [
                self.isotropic.dhistory_rate_dstress(s, hiso, t, erate, T, e)
                + utility.mbmm(
                    self.isotropic.dhistory_rate_derate(s, hiso, t, erate, T, e),
                    derate[..., None, None],
                )[..., 0],
                self.kinematic.dhistory_rate_dstress(s, hkin, t, erate, T, e)
                + utility.mbmm(
                    self.kinematic.dhistory_rate_derate(s, hkin, t, erate, T, e),
                    derate[..., None, None],
                )[..., 0],
            ],
            dim=-1,
        )

    def dhist_derate(self, s, h, t, T, e):
        """
        The derivative of the flow rate with respect to the total strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   internal variables
          t (torch.tensor):   time
          T (torch.tensor):   temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative of flow rate with respect to the strain rate
        """
        erate, _ = self.flow_rate(s, h, t, T, e)
        derate = self.dflow_derate(s, h, t, T, e)

        hiso = h[..., : self.isotropic.nhist]
        hkin = h[..., self.isotropic.nhist :]

        return torch.cat(
            [
                self.isotropic.dhistory_rate_dtotalrate(s, hiso, t, erate, T, e)
                + utility.mbmm(
                    self.isotropic.dhistory_rate_derate(s, hiso, t, erate, T, e),
                    derate[..., None, None],
                )[..., 0],
                self.kinematic.dhistory_rate_dtotalrate(s, hkin, t, erate, T, e)
                + utility.mbmm(
                    self.kinematic.dhistory_rate_derate(s, hkin, t, erate, T, e),
                    derate[..., None, None],
                )[..., 0],
            ],
            dim=-1,
        )
