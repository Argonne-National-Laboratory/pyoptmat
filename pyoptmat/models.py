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

        # Modify for damage
        frate_p = (1 - d) * frate - drate * stress
        dfrate_p = (
            dfrate
            - self.dmodel.d_damage_rate_d_s(stress / (1 - d), d, t, T, erate) / (1 - d)
            - drate
        )

        result = torch.empty_like(y, device=y.device)
        dresult = torch.zeros(y.shape + y.shape[-1:], device=y.device)

        result[..., 0] = self.E(T) * (erate - frate_p)
        result[..., 1 : 1 + self.flowrule.nhist] = hrate
        result[..., -1] = drate

        dresult[..., 0, 0] = -self.E(T) * dfrate_p

        dresult[..., 0:1, 1 : 1 + self.flowrule.nhist] = (
            -self.E(T)[..., None, None]
            * (1 - d)[..., None, None]
            * self.flowrule.dflow_dhist(stress / (1 - d), h, t, T, erate)
        )
        dresult[..., 0, -1] = self.E(T) * (frate - dfrate * stress / (1 - d))

        dresult[..., 1 : 1 + self.flowrule.nhist, 0] = (
            self.flowrule.dhist_dstress(stress / (1 - d), h, t, T, erate)
            / (1 - d)[..., None]
        )
        dresult[..., 1 : 1 + self.flowrule.nhist, 1 : 1 + self.flowrule.nhist] = dhrate
        dresult[..., 1 : 1 + self.flowrule.nhist, -1] = (
            self.flowrule.dhist_dstress(stress / (1 - d), h, t, T, erate)
            * stress[..., None]
            / (1 - d[..., None]) ** 2
        )

        dresult[..., -1, 0] = self.dmodel.d_damage_rate_d_s(
            stress / (1 - d), d, t, T, erate
        ) / (1 - d)
        # d_damage_d_hist is zero
        dresult[..., -1, -1] = ddrate

        # Calculate the derivative wrt the strain rate, used in inverting
        drate = torch.zeros_like(y)
        drate[..., 0] = self.E(T) * (
            1.0
            - (1 - d) * self.flowrule.dflow_derate(stress / (1 - d), h, t, T, erate)
            - self.dmodel.d_damage_rate_d_e(stress / (1 - d), d, t, T, erate) * stress
        )
        drate[..., 1 : 1 + self.flowrule.nhist] = self.flowrule.dhist_derate(
            stress / (1 - d), h, t, T, erate
        )
        drate[..., -1] = self.dmodel.d_damage_rate_d_e(stress / (1 - d), d, t, T, erate)

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
      rtol (optional):              relative tolerance for implicit integration
      atol (optional):              absolute tolerance for implicit integration
      miter (optional):             maximum nonlinear iterations for implicit time integration
      d0 (optional):                intitial value of damage
      use_adjoint (optional):       if `True` use the adjoint approach to
                                    calculate sensitivities, if `False` use
                                    pytorch automatic differentiation
      extra_params (optional):      additional, external parameter to include
                                    in the adjoint calculation.  Used if not
                                    all the parameters can be determined by
                                    introspection
    """

    def __init__(
        self,
        model,
        *args,
        method="backward-euler",
        block_size=1,
        rtol=1.0e-6,
        atol=1.0e-4,
        miter=100,
        d0=0,
        use_adjoint=True,
        extra_params=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.method = method
        self.block_size = block_size
        self.rtol = rtol
        self.atol = atol
        self.miter = miter
        self.d0 = d0
        self.use_adjoint = use_adjoint
        self.extra_params = extra_params

        if self.use_adjoint:
            self.imethod = ode.odeint_adjoint
        else:
            self.imethod = ode.odeint

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

        rate_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(times, rates)
        base_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(times, idata)
        temperature_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            times, temperatures
        )

        init = torch.zeros(
            times.shape[1], 2 + self.model.flowrule.nhist, device=idata.device
        )
        init[:, -1] = self.d0

        bmodel = BothBasedModel(
            self.model,
            rate_interpolator,
            base_interpolator,
            temperature_interpolator,
            control,
        )

        return self.imethod(
            bmodel,
            init,
            times,
            block_size=self.block_size,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            miter=self.miter,
            extra_params=self.extra_params,
        )

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

        init = torch.zeros(
            times.shape[1], 2 + self.model.flowrule.nhist, device=strains.device
        )
        init[:, -1] = self.d0

        emodel = StrainBasedModel(
            self.model, erate_interpolator, temperature_interpolator
        )

        return self.imethod(
            emodel,
            init,
            times,
            block_size=self.block_size,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            miter=self.miter,
            extra_params=self.extra_params,
        )

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

        init = torch.zeros(
            times.shape[1], 2 + self.model.flowrule.nhist, device=stresses.device
        )
        init[:, -1] = self.d0

        smodel = StressBasedModel(
            self.model,
            stress_rate_interpolator,
            stress_interpolator,
            temperature_interpolator,
        )

        return self.imethod(
            smodel,
            init,
            times,
            block_size=self.block_size,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            miter=self.miter,
            extra_params=self.extra_params,
        )

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

    def __init__(self, model, rate_fn, base_fn, T_fn, control, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.rate_fn = rate_fn
        self.base_fn = base_fn
        self.T_fn = T_fn
        self.control = control

        self.emodel = StrainBasedModel(self.model, self.rate_fn, self.T_fn)
        self.smodel = StressBasedModel(
            self.model, self.rate_fn, self.base_fn, self.T_fn
        )

    def forward(self, t, y):
        """
        Evaluate both strain and stress control and paste into the right
        locations.

        Args:
            t:  input times
            y:  input state
        """
        strain_rates, strain_jacs = self.emodel(t, y)
        stress_rates, stress_jacs = self.smodel(t, y)

        actual_rates = torch.zeros_like(strain_rates)

        e_control = self.control == 0
        s_control = self.control == 1

        actual_rates[..., e_control, :] = strain_rates[..., e_control, :]
        actual_rates[..., s_control, :] = stress_rates[..., s_control, :]

        actual_jacs = torch.zeros_like(strain_jacs)
        actual_jacs[..., e_control, :, :] = strain_jacs[..., e_control, :, :]
        actual_jacs[..., s_control, :, :] = stress_jacs[..., s_control, :, :]

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

    def __init__(self, model, srate_fn, stress_fn, T_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.srate_fn = srate_fn
        self.stress_fn = stress_fn
        self.T_fn = T_fn

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

        erate_guess = torch.zeros_like(y[..., 0])[..., None]

        def RJ(erate):
            yp = y.clone()
            yp[..., 0] = cs
            ydot, _, Je, _ = self.model(t, yp, erate[..., 0], cT)

            R = ydot[..., 0] - csr
            J = Je[..., 0]

            return R[..., None], J[..., None, None]

        erate, _ = solvers.newton_raphson(RJ, erate_guess)
        yp = y.clone()
        yp[..., 0] = cs
        ydot, J, Je, _ = self.model(t, yp, erate[..., 0], cT)

        # Rescale the jacobian
        J[..., 0, :] = -J[..., 0, :] / Je[..., 0][..., None]
        J[..., :, 0] = 0

        # Insert the strain rate
        ydot[..., 0] = erate[..., 0]

        return ydot, J
