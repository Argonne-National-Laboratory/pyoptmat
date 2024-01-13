"""
  This module contains objects to define and integrate full material models.

  In our context a material model is a ODE that defines the stress rate and
  an associated set of internal variables.  Mathematically, we can define
  this as two ODEs:

  .. math::

    \\dot{\\sigma} = f(\\sigma, h, T, \\dot{\\varepsilon}, t)

    \\dot{h} = g(\\sigma, h, T, \\dot{\\varepsilon}, t)

  where :math:`\\sigma` is the uniaxial stress, :math:`h` is some
  arbitrary set of internal variables, :math:`T` is the temperature,
  :math:`\\dot{\\varepsilon}` is the strain rate, and :math:`t` is the time.
  Note then that we mathematically define models as strain controlled: the input
  is the strain rate and the output is the stress rate.  Currently there is
  only one implemented full material model: :py:class:`pyoptmat.models.InelasticModel`,
  which is a standard viscoplastic formulation.  However other types of
  models, including rate-independent plasticity, could be defined with
  the same basic form.

  The model itself just defines a system of ODEs.  To solve for stress or
  strain as a function of the experimental conditions we need to integrate
  this model using the methods in :py:mod:`pyoptmat.ode`.  We could do this
  in two ways, in strain control where we provide the strains and temperatures
  as a function of time and integrate for the stress or provide the
  stresses and temperatures as a function of time and integrate for the
  strains.  The :py:class:`pyoptmat.models.ModelIntegrator` provides both
  options, where each experiment can either be strain or stress controlled.

  The basic process of setting up a material model capable of simulating
  experimental tests is to define the model form mathematically, using a
  Model class and wrap that Model with an Integrator to provide actual
  time series of stress or strain.  As the integrator class uses the
  methods in :py:mod:`pyoptmat.ode` to actually do the integration, the
  results (and subsequent mathematical operations on the results) can be
  differentiated using either PyTorch backpropogation AD or the adjoint
  method.
"""

import torch
from torch import nn

from pyoptmat import utility, ode, damage, solvers


class InelasticModel(nn.Module):
    """
    This object provides the standard strain-based rate form of a constitutive model

    .. math::

      \\dot{\\sigma} = E \\left(\\dot{\\varepsilon} - \\dot{\\varepsilon}_{in} \\right)

    Args:
      E:                        Material Young's modulus
      flowrule:                 :py:class:`pyoptmat.flowrules.FlowRule` defining the inelastic
                                strain rate
    """

    def __init__(self, E, flowrule, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.E = E
        self.flowrule = flowrule

    @property
    def nhist(self):
        """
        Number of internal variables
        """
        return 1 + self.flowrule.nhist

    def forward(self, t, y, erate, T):
        """
        Return the rate equations for the strain-based version of the model

        Args:
          t:              (nbatch,) times
          y:              (nbatch,1+nhist) [stress, history]
          erate:          (nbatch,) strain rates
          T:              (nbatch,) temperatures

        Returns:
          y_dot:          (nbatch,1+nhist) state rate
          d_y_dot_d_y:    (nbatch,1+nhist,1+nhist) Jacobian wrt the state
          d_y_dot_d_erate:(nbatch,1+nhist) Jacobian wrt the strain rate
          d_y_dot_d_T:    (nbatch,1+nhist) derivative wrt temperature (unused)
        """
        stress = y[..., 0]
        h = y[..., 1:]

        frate, dfrate = self.flowrule.flow_rate(stress, h, t, T, erate)
        hrate, dhrate = self.flowrule.history_rate(stress, h, t, T, erate)

        # Stacked rate of evolution vector
        result = torch.cat(
            [(self.E(T) * (erate - frate)).unsqueeze(-1), hrate],
            dim=-1,
        )

        # Form the large blocked matrix of d(y_dot)/d(y)
        row1 = torch.cat(
            [
                (-self.E(T) * dfrate).unsqueeze(-1).unsqueeze(-1),
                (
                    -self.E(T)[..., None, None]
                    * self.flowrule.dflow_dhist(stress, h, t, T, erate)
                ),
            ],
            dim=-1,
        )

        row2 = torch.cat(
            [self.flowrule.dhist_dstress(stress, h, t, T, erate).unsqueeze(-1), dhrate],
            dim=-1,
        )
        dresult = torch.cat([row1, row2], dim=-2)

        # Form the stacked derivative of the state rate with respect to the strain rate
        drate = torch.cat(
            [
                (
                    self.E(T)
                    * (1.0 - self.flowrule.dflow_derate(stress, h, t, T, erate))
                ).unsqueeze(-1),
                self.flowrule.dhist_derate(stress, h, t, T, erate),
            ],
            dim=-1,
        )

        # Logically we should return the derivative wrt T, but right now
        # we're not going to use it
        Trate = torch.zeros_like(y)

        return result, dresult, drate, Trate


class DamagedInelasticModel(nn.Module):
    """
    This object provides the standard strain-based rate form of a constitutive model

    .. math::

      \\dot{\\sigma} = E \\left(\\dot{\\varepsilon} - (1-d) \\dot{\\varepsilon}_{in} \\right)

    Args:
      E:                        Material Young's modulus
      flowrule:                 :py:class:`pyoptmat.flowrules.FlowRule` defining the inelastic
                                strain rate
      dmodel (optional):        :py:class:`pyoptmat.damage.DamageModel` defining the damage variable
                                  evolution rate, defaults to :py:class:`pyoptmat.damage.NoDamage`
    """

    def __init__(self, E, flowrule, *args, dmodel=damage.NoDamage(), **kwargs):
        super().__init__(*args, **kwargs)
        self.E = E
        self.flowrule = flowrule
        self.dmodel = dmodel

    @property
    def nhist(self):
        """
        Number of internal variables
        """
        return 2 + self.flowrule.nhist

    def forward(self, t, y, erate, T):
        """
        Return the rate equations for the strain-based version of the model

        Args:
          t:              (nbatch,) times
          y:              (nbatch,1+nhist+1) [stress, history, damage]
          erate:          (nbatch,) strain rates
          T:              (nbatch,) temperatures

        Returns:
          y_dot:          (nbatch,1+nhist+1) state rate
          d_y_dot_d_y:    (nbatch,1+nhist+1,1+nhist+1) Jacobian wrt the state
          d_y_dot_d_erate:(nbatch,1+nhist+1) Jacobian wrt the strain rate
          d_y_dot_d_T:    (nbatch,1+nhist+1) derivative wrt temperature (unused)
        """
        stress = y[..., 0].clone()
        h = y[..., 1 : 1 + self.flowrule.nhist].clone()
        d = y[..., -1].clone()

        frate, dfrate = self.flowrule.flow_rate(stress / (1 - d), h, t, T, erate)
        hrate, dhrate = self.flowrule.history_rate(stress / (1 - d), h, t, T, erate)
        drate, ddrate = self.dmodel.damage_rate(stress / (1 - d), d, t, T, erate)

        # Stacked rate of evolution vector
        result = torch.cat(
            [
                ((1 - d) * self.E(T) * (erate - frate)).unsqueeze(-1),
                hrate,
                drate.unsqueeze(-1),
            ],
            dim=-1,
        )

        # Form the large blocked matrix of d(y_dot)/d(y)
        row1 = torch.cat(
            [
                (-self.E(T) * dfrate).unsqueeze(-1).unsqueeze(-1),
                (
                    -self.E(T)[..., None, None]
                    * self.flowrule.dflow_dhist(stress / (1 - d), h, t, T, erate)
                    * (1 - d)[..., None, None]
                ),
                (-self.E(T) * (erate - frate) - self.E(T) * dfrate * stress / (1 - d))
                .unsqueeze(-1)
                .unsqueeze(-1),
            ],
            dim=-1,
        )
        row2 = torch.cat(
            [
                (
                    self.flowrule.dhist_dstress(stress / (1 - d), h, t, T, erate)
                    / (1 - d)[..., None]
                ).unsqueeze(-1),
                dhrate,
                (
                    self.flowrule.dhist_dstress(stress / (1 - d), h, t, T, erate)
                    * stress[..., None]
                    / (1 - d[..., None]) ** 2
                ).unsqueeze(-1),
            ],
            dim=-1,
        )
        row3 = torch.cat(
            [
                (
                    self.dmodel.d_damage_rate_d_s(stress / (1 - d), d, t, T, erate)
                    / (1 - d)
                )[..., None, None],
                torch.zeros_like(h).unsqueeze(-2),
                ddrate[..., None, None],
            ],
            dim=-1,
        )
        dresult = torch.cat([row1, row2, row3], dim=-2)

        # Form the stacked derivative of the state rate with respect to the strain rate
        drate = torch.cat(
            [
                (
                    self.E(T)
                    * (1 - d)
                    * (
                        1.0
                        - self.flowrule.dflow_derate(stress / (1 - d), h, t, T, erate)
                    )
                ).unsqueeze(-1),
                self.flowrule.dhist_derate(stress / (1 - d), h, t, T, erate),
                self.dmodel.d_damage_rate_d_e(
                    stress / (1 - d), d, t, T, erate
                ).unsqueeze(-1),
            ],
            dim=-1,
        )

        # Logically we should return the derivative wrt T, but right now
        # we're not going to use it
        Trate = torch.zeros_like(y)

        return result, dresult, drate, Trate


class ModelIntegrator(nn.Module):
    """
    This class provides infrastructure for integrating constitutive models in
    either strain or stress control.

    Args:
      model:                        base strain-controlled model
      method (optional):            integrate method used to solve the equations, defaults
                                    to `"backward-euler"`
      use_adjoint (optional):       if `True` use the adjoint approach to
      **kwargs:                     passed on to the odeint method

    """

    def __init__(self, model, *args, use_adjoint=True, bisect_first=False, throw_on_scalar_fail = False, **kwargs):
        super().__init__(*args)
        self.model = model
        self.use_adjoint = use_adjoint
        self.throw_on_scalar_fail = throw_on_scalar_fail
        self.kwargs_for_integration = kwargs

        if self.use_adjoint:
            self.imethod = ode.odeint_adjoint
        else:
            self.imethod = ode.odeint

        self.bisect_first = bisect_first

    def solve_both(self, times, temperatures, idata, control):
        """
        Solve for either strain or stress control at once

        Args:
          times:          input times, (ntime,nexp)
          temperatures:   input temperatures (ntime,nexp)
          idata:          input data (ntime,nexp)
          control:        signal for stress/strain control (nexp,)
        """
        rates = torch.cat(
            (
                torch.zeros(1, idata.shape[1], device=idata.device),
                (idata[1:] - idata[:-1]) / (times[1:] - times[:-1]),
            )
        )
        # Likely if this happens dt = 0
        rates[torch.isnan(rates)] = 0

        init = torch.zeros(times.shape[1], self.model.nhist, device=idata.device)

        bmodel = BothBasedModel(
            self.model,
            times,
            rates,
            idata,
            temperatures,
            control,
            bisect_first=self.bisect_first,
            throw_on_scalar_fail = self.throw_on_scalar_fail
        )

        return self.imethod(bmodel, init, times, **self.kwargs_for_integration)

    def solve_strain(self, times, strains, temperatures):
        """
        Basic model definition: take time and strain rate and return stress

        Args:
          times:          input times, shape (ntime)
          strains:        input strains, shape (ntime, nbatch)
          temperatures:   input temperatures, shape (ntime, nbatch)

        Returns:
          y:          stacked [stress, history, damage] vector of shape
                      `(ntime,nbatch,1+nhist+1)`
        """
        strain_rates = torch.cat(
            (
                torch.zeros(1, strains.shape[1], device=strains.device),
                (strains[1:] - strains[:-1]) / (times[1:] - times[:-1]),
            )
        )
        # Likely if this happens dt = 0
        strain_rates[torch.isnan(strain_rates)] = 0

        erate_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            times, strain_rates
        )
        temperature_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            times, temperatures
        )

        init = torch.zeros(times.shape[1], self.model.nhist, device=strains.device)

        emodel = StrainBasedModel(
            self.model, erate_interpolator, temperature_interpolator
        )

        return self.imethod(emodel, init, times, **self.kwargs_for_integration)

    def solve_stress(self, times, stresses, temperatures):
        """
        Inverse model definition: take time and stress rate and return strain

        Args:
          times:          input times, shape (ntime,)
          stresses:       input stresses, shape (ntime, nbatch)
          temperatures:   input temperatures, shape (ntime, nbatch)

        Returns:
          y:              stack [strain, history, damage] vector
                          of shape `(ntime,nbatch,2+nhist)`
        """
        stress_rates = torch.cat(
            (
                torch.zeros(1, stresses.shape[1], device=stresses.device),
                (stresses[1:] - stresses[:-1]) / (times[1:] - times[:-1]),
            )
        )
        # Likely if this happens dt = 0
        stress_rates[torch.isnan(stress_rates)] = 0

        stress_rate_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            times, stress_rates
        )
        stress_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            times, stresses
        )
        temperature_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            times, temperatures
        )

        init = torch.zeros(times.shape[1], self.model.nhist, device=stresses.device)

        smodel = StressBasedModel(
            self.model,
            stress_rate_interpolator,
            stress_interpolator,
            temperature_interpolator,
            bisect_first = self.bisect_first,
            throw_on_scalar_fail = self.throw_on_scalar_fail
        )

        return self.imethod(smodel, init, times, **self.kwargs_for_integration)

    def forward(self, t, y):
        """
        Evaluate both strain and stress control and paste into the right
        locations.

        Args:
            t:  input times
            y:  input state
        """
        raise NotImplementedError("forward method is pure virtual in base class")


class BothBasedModel(nn.Module):
    """
    Provides both the strain rate and stress rate form at once, for better vectorization

    Args:
      model:    base InelasticModel
      rate_fn:  controlled quantity rate interpolator
      base_fn:  controlled quantity base interpolator
      T_fn:     temperature interpolator
      indices:  split into strain and stress control
    """

    def __init__(
        self,
        model,
        times,
        rates,
        base,
        temps,
        control,
        bisect_first=False,
        throw_on_scalar_fail = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.control = control

        self.econtrol = self.control == 0
        self.scontrol = self.control == 1

        self.emodel = StrainBasedModel(
            self.model,
            utility.ArbitraryBatchTimeSeriesInterpolator(
                times[..., self.econtrol], rates[..., self.econtrol]
            ),
            utility.ArbitraryBatchTimeSeriesInterpolator(
                times[..., self.econtrol], temps[..., self.econtrol]
            ),
        )
        self.smodel = StressBasedModel(
            self.model,
            utility.ArbitraryBatchTimeSeriesInterpolator(
                times[..., self.scontrol], rates[..., self.scontrol]
            ),
            utility.ArbitraryBatchTimeSeriesInterpolator(
                times[..., self.scontrol], base[..., self.scontrol]
            ),
            utility.ArbitraryBatchTimeSeriesInterpolator(
                times[..., self.scontrol], temps[..., self.scontrol]
            ),
            bisect_first=bisect_first,
            throw_on_scalar_fail = throw_on_scalar_fail
        )

    def forward(self, t, y):
        """
        Evaluate both strain and stress control and paste into the right
        locations.

        Args:
            t:  input times
            y:  input state
        """
        n = (y.shape[-1],)
        base = y.shape[:-1]

        actual_rates = torch.zeros(base + n, device=t.device)
        actual_jacs = torch.zeros(base + n + n, device=t.device)

        if torch.any(self.econtrol):
            strain_rates, strain_jacs = self.emodel(
                t[..., self.econtrol], y[..., self.econtrol, :]
            )
            actual_rates[..., self.econtrol, :] = strain_rates
            actual_jacs[..., self.econtrol, :, :] = strain_jacs

        if torch.any(self.scontrol):
            stress_rates, stress_jacs = self.smodel(
                t[..., self.scontrol], y[..., self.scontrol, :]
            )
            actual_rates[..., self.scontrol, :] = stress_rates
            actual_jacs[..., self.scontrol, :, :] = stress_jacs

        return actual_rates, actual_jacs


class StrainBasedModel(nn.Module):
    """
    Provides the strain rate form

    Args:
      model:        base InelasticModel
      erate_fn:     erate(t)
      T_fn:         T(t)
    """

    def __init__(self, model, erate_fn, T_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.erate_fn = erate_fn
        self.T_fn = T_fn

    def forward(self, t, y):
        """
        Strain rate as a function of t and state

        Args:
            t:  input times
            y:  input state
        """
        return self.model(t, y, self.erate_fn(t), self.T_fn(t))[
            :2
        ]  # Don't need the extras


class StressBasedModel(nn.Module):
    """
    Provides the stress rate form

    Args:
      model:        base InelasticModel
      srate_fn:     srate(t)
      T_fn:         T(t)
    """

    def __init__(
        self,
        model,
        srate_fn,
        stress_fn,
        T_fn,
        min_erate=-1e2,
        max_erate=1e3,
        guess_erate=1.0e-3,
        bisect_first=False,
        throw_on_scalar_fail = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.srate_fn = srate_fn
        self.stress_fn = stress_fn
        self.T_fn = T_fn
        self.min_erate = min_erate
        self.max_erate = max_erate
        self.bisect_first = bisect_first
        self.guess_erate = guess_erate
        self.throw_on_scalar_fail = throw_on_scalar_fail

    def forward(self, t, y):
        """
        Stress rate as a function of t and state

        Args:
            t:  input times
            y:  input state
        """
        csr = self.srate_fn(t)
        cs = self.stress_fn(t)
        cT = self.T_fn(t)

        def RJ(erate):
            ydot, _, Je, _ = self.model(
                t, torch.cat([cs.unsqueeze(-1), y[..., 1:]], dim=-1), erate, cT
            )

            R = ydot[..., 0] - csr
            J = Je[..., 0]

            return R, J

        # Doing the detach here actually makes the parameter gradients wrong...
        if self.bisect_first:
            erate = solvers.scalar_bisection_newton(
                RJ,
                torch.ones_like(y[..., 0]) * self.min_erate,
                torch.ones_like(y[..., 0]) * self.max_erate,
                throw_on_fail = self.throw_on_scalar_fail
            )
        else:
            erate = solvers.scalar_newton(RJ, torch.sign(csr) * self.guess_erate,
                    throw_on_fail = self.throw_on_scalar_fail)

        ydot, J, Je, _ = self.model(
            t, torch.cat([cs.unsqueeze(-1), y[..., 1:]], dim=-1), erate, cT
        )

        # There is an annoying extra term that is the derivative of the history rate with respect to the
        # solved for strain rate times the derivative of the strain rate with respect to history
        t1 = Je[..., 1:].unsqueeze(-1)
        t2 = utility.mbmm(1.0 / Je[..., :1].unsqueeze(-1), J[..., 0, 1:].unsqueeze(-2))
        t3 = utility.mbmm(t1, t2)

        # Corrected jacobian
        row1 = torch.cat(
            [
                torch.zeros_like(J[..., 0, 0]).unsqueeze(-1),
                -J[..., 0, 1:] / Je[..., 0][..., None],
            ],
            dim=-1,
        ).unsqueeze(-2)
        rest = torch.cat(
            [torch.zeros_like(J[..., 1:, 0]).unsqueeze(-1), J[..., 1:, 1:] - t3], dim=-1
        )
        jac = torch.cat([row1, rest], dim=-2)

        return torch.cat([erate.unsqueeze(-1), ydot[..., 1:]], dim=-1), jac
