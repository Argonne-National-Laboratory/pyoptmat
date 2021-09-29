import torch
import torch.nn as nn

from pyoptmat import utility, ode, damage

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

    self._setup(t, strain_rates, temperatures)

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

  def _setup(self, t, strain_rates, temperatures):
    """
      Setup before a solve.  This gets called after sampling parameters,
      for inference problems

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

