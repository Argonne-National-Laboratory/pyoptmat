import torch
import torch.nn as nn

from pyoptmat import utility, ode, damage, solvers

class InelasticModel(nn.Module):
  """
    This object provides the basic response used to run and calibrate 
    constitutive models: an inelastic model defined by a flowrule and the
    standard viscoplastic stress rate equation

    .. math::

      \\dot{\\sigma} = E \\left(\\dot{\\varepsilon} - (1-d) \\dot{\\varepsilon}_{in} \\right)

    Args:
      E:                        Material Young's modulus
      flowrule:                 :py:class:`pyoptmat.flowrules.FlowRule` defining the inelastic
                                strain rate
      substeps (optional):      subdivide each provided timestep into multiple steps to
                                reduce integration error, defaults to 1
      method (optional):        integrate method used to solve the equations, defaults 
                                to `"backward-euler"`
      dmodel (optional):        :py:class:`pyoptmat.damage.DamageModel` defining the damage variable
                                evolution rate, defaults to :py:class:`pyoptmat.damage.NoDamage`
      rtol (optional):          relative tolerance for implicit integration
      atol (optional):          absolute tolerance for implicit integration
      progress (optional):      print a progress bar for forward time integration
      miter (optional):         maximum nonlinear iterations for implicit time integration
      d0 (optional):            intitial value of damage
      use_adjoint (optional):       if `True` use the adjoint approach to 
                                    calculate sensitivities, if `False` use 
                                    pytorch automatic differentiation
      extra_params (optional):      additional, external parameter to include
                                    in the adjoint calculation.  Used if not
                                    all the parameters can be determined by
                                    introspection
      jit_mode (optional):          if true use the JIT mode which cuts out 
                                    error checking and fixes sizes
  """
  def __init__(self, E, flowrule, substeps = 1, method = 'backward-euler', 
      dmodel = damage.NoDamage(),
      rtol = 1.0e-6, atol = 1.0e-4, progress = False, 
      miter = 100, d0 = 0, use_adjoint = True, extra_params = [],
      jit_mode = False):
    super().__init__()
    self.E = E
    self.flowrule = flowrule
    self.dmodel = dmodel
    self.times = torch.zeros(1,1)
    self.strain_rates = torch.zeros(1,1)
    self.substeps = substeps
    self.method = method
    self.rtol = rtol
    self.atol = atol
    self.progress = progress
    self.miter = miter
    self.d0 = d0
    self.use_adjoint = use_adjoint
    self.extra_params = extra_params
    self.jit_mode = jit_mode

  def solve(self, t, strains, temperatures):
    """
      Basic model definition: take time and strain rate and return stress

      Args:
        t:              input times, shape (ntime)
        strains:        input strains, shape (ntime, nbatch)
        temperatuers:   input temperatuers, shape (ntime, nbatch)

      Returns:
        y:          stacked [stress, history, damage] vector of shape 
                    `(ntime,nbatch,1+nhist+1)`
    """
    device = strains.device
    strain_rates = torch.cat((torch.zeros(1,strains.shape[1], device = device),
      (strains[1:]-strains[:-1])/(t[1:]-t[:-1])))
    # Likely if this happens dt = 0
    strain_rates[torch.isnan(strain_rates)] = 0

    self._setup_strain_control(t, strain_rates, temperatures)

    init = torch.zeros(self.nbatch, self.nsize, device = strains.device)
    init[:,-1] = self.d0
    
    if self.use_adjoint:
      imethod = ode.odeint_adjoint
    else:
      imethod = ode.odeint

    return imethod(self, init,  t,
        method = self.method, substep = self.substeps, rtol = self.rtol, 
        atol = self.atol, progress = self.progress, miter = self.miter,
        extra_params = self.extra_params, jit_mode = self.jit_mode)

  def _setup_strain_control(self, t, strain_rates, temperatures):
    """
      Setup before a strain-controlled solve.  

      Args:
        t:              input times
        strain_rates:   input strain rates
        temperatures:   input temperatures
    """
    self.nbatch = t.shape[1]
    self.nsize = 1 + self.flowrule.nhist + 1 # Stress + history + damage
    self.device = strain_rates.device
    
    self.times = t
    self.strain_rates = strain_rates
    self.temperatures = temperatures
    
    self.erate_interpolator = utility.CheaterBatchTimeSeriesInterpolator(
        self.times, self.strain_rates)
    self.temperature_interpolator = utility.CheaterBatchTimeSeriesInterpolator(
        self.times, self.temperatures)

  def forward(self, t, states):
    """
      Forward model call function: returns the rate of [stress, history, damage] 
      and the Jacobian

      Args:
        t:          input time, shape (nbatch,)
        states:     input state, shape (nbatch,1+nhist+1)

      Returns:
        y_dot:      state rate, shape (nbatch,1+nhist+1)
        jacobian:   d_y_dot/d_y, shape (nbatch,1+nhist+1,1+nhist+1)
    """
    stress = states[:,0].clone()
    h = states[:,1:1+self.flowrule.nhist].clone()
    d = states[:,-1].clone()

    erate = self.erate_interpolator(t)
    T = self.temperature_interpolator(t)
    
    frate, dfrate = self.flowrule.flow_rate(stress/(1-d), h, t, T)
    hrate, dhrate = self.flowrule.history_rate(stress/(1-d), h, t, T)
    drate, ddrate = self.dmodel.damage_rate(stress/(1-d), d, t, T)

    # Modify for damage
    frate_p = (1-d)*frate - drate * stress
    dfrate_p = dfrate - self.dmodel.d_damage_rate_d_s(
        stress/(1-d), d, t, T) / (1-d) - drate

    result = torch.empty_like(states, device = states.device)
    dresult = torch.zeros(states.shape + states.shape[1:], 
        device = states.device)

    result[:,0] = self.E(T) * (erate - frate_p)
    result[:,1:1+self.flowrule.nhist] = hrate
    result[:,-1] = drate
    
    dresult[:,0,0] = -self.E(T) * dfrate_p

    dresult[:,0:1,1:1+self.flowrule.nhist] = -self.E(T)[...,None,None] * (1-d)[:,None,None
        ] * self.flowrule.dflow_dhist(stress/(1-d),h, t, T)
    dresult[:,0,-1] = self.E(T) * (frate - dfrate * stress/(1-d))
    
    dresult[:,1:1+self.flowrule.nhist,0] = self.flowrule.dhist_dstress(
        stress/(1-d), h, t, T) / (1-d)[:,None]
    dresult[:,1:1+self.flowrule.nhist,1:1+self.flowrule.nhist] = dhrate
    dresult[:,1:1+self.flowrule.nhist,-1] = self.flowrule.dhist_dstress(
        stress/(1-d), h, t, T) * stress[:,None] / (1-d[:,None])**2

    dresult[:,-1,0] = self.dmodel.d_damage_rate_d_s(stress/(1-d), d, t, T) / (1-d)
    # d_damage_d_hist is zero
    dresult[:,-1,-1] = ddrate

    return result, dresult


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
  def __init__(self, E, flowrule, *args, dmodel = damage.NoDamage(), **kwargs):
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
    """
    stress = y[:,0].clone()
    h = y[:,1:1+self.flowrule.nhist].clone()
    d = y[:,-1].clone()

    frate, dfrate = self.flowrule.flow_rate(stress/(1-d), h, t, T)
    hrate, dhrate = self.flowrule.history_rate(stress/(1-d), h, t, T)
    drate, ddrate = self.dmodel.damage_rate(stress/(1-d), d, t, T)

    # Modify for damage
    frate_p = (1-d)*frate - drate * stress
    dfrate_p = dfrate - self.dmodel.d_damage_rate_d_s(
        stress/(1-d), d, t, T) / (1-d) - drate

    result = torch.empty_like(y, device = y.device)
    dresult = torch.zeros(y.shape + y.shape[1:], device = y.device)

    result[:,0] = self.E(T) * (erate - frate_p)
    result[:,1:1+self.flowrule.nhist] = hrate
    result[:,-1] = drate
    
    dresult[:,0,0] = -self.E(T) * dfrate_p

    dresult[:,0:1,1:1+self.flowrule.nhist] = -self.E(T)[...,None,None] * (1-d)[:,None,None
        ] * self.flowrule.dflow_dhist(stress/(1-d),h, t, T)
    dresult[:,0,-1] = self.E(T) * (frate - dfrate * stress/(1-d))
    
    dresult[:,1:1+self.flowrule.nhist,0] = self.flowrule.dhist_dstress(
        stress/(1-d), h, t, T) / (1-d)[:,None]
    dresult[:,1:1+self.flowrule.nhist,1:1+self.flowrule.nhist] = dhrate
    dresult[:,1:1+self.flowrule.nhist,-1] = self.flowrule.dhist_dstress(
        stress/(1-d), h, t, T) * stress[:,None] / (1-d[:,None])**2

    dresult[:,-1,0] = self.dmodel.d_damage_rate_d_s(stress/(1-d), d, t, T
        ) / (1-d)
    # d_damage_d_hist is zero
    dresult[:,-1,-1] = ddrate

    # Calculate the derivative wrt the strain rate, used in inverting
    drate = torch.zeros_like(y)
    drate[:,0] = self.E(T)

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
      substeps (optional):          subdivide each provided timestep into multiple steps to
                                    reduce integration error, defaults to 1
      method (optional):            integrate method used to solve the equations, defaults 
                                    to `"backward-euler"`
      rtol (optional):              relative tolerance for implicit integration
      atol (optional):              absolute tolerance for implicit integration
      progress (optional):          print a progress bar for forward time integration
      miter (optional):             maximum nonlinear iterations for implicit time integration
      d0 (optional):                intitial value of damage
      use_adjoint (optional):       if `True` use the adjoint approach to 
                                    calculate sensitivities, if `False` use 
                                    pytorch automatic differentiation
      extra_params (optional):      additional, external parameter to include
                                    in the adjoint calculation.  Used if not
                                    all the parameters can be determined by
                                    introspection
      jit_mode (optional):          if true use the JIT mode which cuts out 
                                    error checking and fixes sizes
  """
  def __init__(self, model, *args, substeps = 1, method = 'backward-euler', 
      rtol = 1.0e-6, atol = 1.0e-4, progress = False, 
      miter = 100, d0 = 0, use_adjoint = True, extra_params = [],
      jit_mode = False, **kwargs):
    super().__init__(*args, **kwargs)
    self.model = model
    self.substeps = substeps
    self.method = method
    self.rtol = rtol
    self.atol = atol
    self.progress = progress
    self.miter = miter
    self.d0 = d0
    self.use_adjoint = use_adjoint
    self.extra_params = extra_params
    self.jit_mode = jit_mode

    if self.use_adjoint:
      self.imethod = ode.odeint_adjoint
    else:
      self.imethod = ode.odeint

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
    strain_rates = torch.cat((torch.zeros(1,strains.shape[1], device = strains.device),
      (strains[1:]-strains[:-1])/(times[1:]-times[:-1])))
    # Likely if this happens dt = 0
    strain_rates[torch.isnan(strain_rates)] = 0

    erate_interpolator = utility.CheaterBatchTimeSeriesInterpolator(
        times, strain_rates)
    temperature_interpolator = utility.CheaterBatchTimeSeriesInterpolator(
        times, temperatures)

    init = torch.zeros(times.shape[1], 2 + self.model.flowrule.nhist,
        device = strains.device)
    init[:,-1] = self.d0

    emodel = StrainBasedModel(self.model, erate_interpolator, temperature_interpolator)

    return self.imethod(emodel,
      init, times, method = self.method, substep = self.substeps, rtol = self.rtol, 
      atol = self.atol, progress = self.progress, miter = self.miter,
      extra_params = self.extra_params, jit_mode = self.jit_mode)

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
    stress_rates = torch.cat((torch.zeros(1,stresses.shape[1], device = stresses.device),
      (stresses[1:]-stresses[:-1])/(times[1:]-times[:-1])))
    # Likely if this happens dt = 0
    stress_rates[torch.isnan(stress_rates)] = 0

    stress_rate_interpolator = utility.CheaterBatchTimeSeriesInterpolator(
        times, stress_rates)
    stress_interpolator = utility.CheaterBatchTimeSeriesInterpolator(
        times, stresses)
    temperature_interpolator = utility.CheaterBatchTimeSeriesInterpolator(
        times, temperatures)

    init = torch.zeros(times.shape[1], 2 + self.model.flowrule.nhist,
        device = stresses.device)
    init[:,-1] = self.d0

    smodel = StressBasedModel(self.model, stress_rate_interpolator, stress_interpolator,
        temperature_interpolator)

    return self.imethod(smodel,
      init, times, method = self.method, substep = self.substeps, rtol = self.rtol, 
      atol = self.atol, progress = self.progress, miter = self.miter,
      extra_params = self.extra_params, jit_mode = self.jit_mode)

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
    """
    return self.model(t, y, self.erate_fn(t), self.T_fn(t))[:2] # Don't need the extras

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
        t:      current batch time
        y:      current batch state
    """
    csr = self.srate_fn(t)
    cs = self.stress_fn(t)
    cT = self.T_fn(t)

    erate_guess = torch.zeros_like(y[:,0])[:,None]

    def RJ(erate):
      yp = y.clone()
      yp[:,0] = cs
      ydot, Js, Je, JT = self.model(t, yp, erate[:,0], cT)

      R = ydot[:,0] - csr
      J = Je[:,0]

      return R[:,None], J[:,None,None]
    
    erate, _ = solvers.newton_raphson(RJ, erate_guess)
    yp = y.clone()
    yp[:,0] = cs
    ydot, J, Je, JT = self.model(t, yp, erate[:,0], cT)

    # Rescale the jacobian
    J[:,0,:] = -J[:,0,:] / Je[:,0][:,None]
    J[:,:,0] = 0
    
    # Insert the strain rate
    ydot[:,0] = erate[:,0]

    return ydot, J
