"""
  Objects and helper functions to help with deterministic model calibration
  and statistical inference.
"""

from collections import defaultdict
import warnings

import numpy as np
import scipy.interpolate as inter
import scipy.optimize as opt

import torch
from torch.nn import Module, Parameter

from skopt.space import Space
from skopt.sampler import Lhs, Halton

import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.contrib.autoguide import AutoDelta, init_to_mean

from tqdm import tqdm

from pyoptmat import experiments

def bound_factor(mean, factor, min_bound = None):
  """
    Apply the bounded_scale_function map but set the upper bound as mean*(1+factor)
    and the lower bound as mean*(1-factor)

    Args:
      mean:         center value
      factor:       bounds factor

    Additional Args:
      min_bound:    clip to avoid going lower than this value
  """
  return bounded_scale_function((mean*(1.0-factor),mean*(1+factor)), min_bound = min_bound)

def bounded_scale_function(bounds, min_bound = None):
  """
    Sets up a scaling function that maps `(0,1)` to `(bounds[0], bounds[1])`
    and clips the values to remain in that range

    Args:
      bounds:   tuple giving the parameter bounds

    Additional Args:
      min_bounds:   clip to avoid going lower than this value
  """
  if min_bound is None:
    return lambda x: torch.clamp(x, 0, 1)*(bounds[1]-bounds[0]) + bounds[0]
  else:
    return lambda x: torch.maximum(torch.clamp(x, 0, 1)*(bounds[1]-bounds[0]) + bounds[0], min_bound)

def clamp_scale_function(bounds):
  """
    Just clamp to bounds

    Args:
      bounds:   tuple giving the parameter bounds
  """
  return lambda x: torch.clamp(x, bounds[0], bounds[1])

def bound_and_scale(scale, bounds):
  """
    Combination of scaling and bounding

    Args:
      scale:    divide input by this value
      bounds:   tuple, clamp to (bounds[0], bounds[1])
  """
  return lambda x: torch.clamp(x / scale, bounds[0], bounds[1])

def log_bound_and_scale(scale, bounds):
  """
    Scale, de-log, and bound

    Args:
      scale:    divide input by this value, then take exp
      bounds:   tuple, clamp to (bounds[0], bounds[1])
  """
  return lambda x: torch.clamp(torch.exp(x / scale), bounds[0], bounds[1])

class DeterministicModelExperiment(Module):
  """
    Wrap a material model to provide a :py:mod:`pytorch` deterministic model

    Args:
      maker:      function that returns a valid model as a  Module, given the 
                  input parameters
      names:      names to use for the parameters
      ics:        initial conditions to use for each parameter
  """
  def __init__(self, maker, names, ics):
    super().__init__()

    self.maker = maker

    self.names = names
    
    # Add all the parameters
    self.params = names
    for name, ic in zip(names, ics):
      setattr(self, name, Parameter(ic))
    
  def get_params(self):
    """
      Return the parameters for input to the model
    """
    return [getattr(self, name) for name in self.params]

  def forward(self, exp_data, exp_cycles, exp_types, exp_control):
    """
      Integrate forward and return the results

      Args:
        exp_data:       formatted input experimental data (see experiments module)
        exp_cycles:     cycle counts for each test
        exp_types:      experiment types, as integers
        exp_control:    stress/strain control flag
    """
    model = self.maker(*self.get_params())

    predictions = model.solve_both(exp_data[0], exp_data[1],
        exp_data[2], exp_control)

    return experiments.convert_results(predictions[:,:,0], exp_cycles, exp_types)

class StatisticalModelExperiment(PyroModule):
  """
    Wrap a material model to provide a py:mod:`pyro` single-level 
    statistical model

    Single level means each parameter is sampled once before running
    all the tests -- i.e. each test is run on the "heat" of material

    Args:
      maker:      function that returns a valid Module, given the 
                  input parameters
      names:      names to use for the parameters
      loc:        parameter location priors
      scales:     parameter scale priors
      eps:        random noise, can be either a single scalar or a 1D tensor
                  if it's a 1D tensor then each entry i represents the
                  noise in test type i
  """
  def __init__(self, maker, names, locs, scales, eps):
    super().__init__()

    self.maker = maker
    
    # Add all the parameters
    self.params = names
    for name, loc, scale in zip(names, locs, scales):
      setattr(self, name, PyroSample(dist.Normal(loc, scale)))

    self.eps = eps

    self.type_noise = self.eps.dim() > 0

  def get_params(self):
    """
      Return the sampled parameters for input to the model
    """
    return [getattr(self, name) for name in self.params]

  def forward(self, exp_data, exp_cycles, exp_types, exp_control, exp_results = None):
    """
      Integrate forward and return the result

      Optionally condition on the actual data

      Args:
        exp_data:       formatted experimental data, see experiments module
        exp_cycles:     cycle counts for each test
        exp_types:      experiment types for each test
        exp_control:    stress/strain control flag for each test

      Additional Args:
        exp_results:    true results for conditioning (ntime,nexp)
    """
    model = self.maker(*self.get_params())
    predictions = model.solve_both(exp_data[0], exp_data[1], exp_data[2], exp_control)
    results = experiments.convert_results(predictions[:,:,0], exp_cycles, exp_types)

    # Setup the full noise, which can be type specific
    if self.type_noise:
      full_noise = torch.empty(exp_data.shape[-1])
      for i in experiments.exp_map.values():
        full_noise[exp_types==i] = self.eps[i]
    else:
      full_noise = self.eps

    with pyro.plate("trials", exp_data.shape[2]):
      with pyro.plate("time", exp_data.shape[1]):
        return pyro.sample("obs", dist.Normal(results, full_noise), obs = exp_results)

class HierarchicalStatisticalModelExperiment(PyroModule):
  """
    Wrap a material model to provide a hierarchical :py:mod:`pyro` statistical
    model

    This type of statistical model does two levels of sampling for each
    parameter in the base model.

    First it samples a random variable to select the mean and scale of the
    population of parameter values

    Then, based on this "top level" location and scale it samples each parameter
    independently -- i.e. each experiment is drawn from a different "heat",
    with population statistics given by the top samples

    At least for the moment the population means are selected from 
    normal distributions, the population standard deviations from HalfNormal
    distributions, and then each parameter population comes from a
    Normal distribution

    Args: 
      maker:                    function that returns a valid material Module, 
                                given the input parameters
      names:                    names to use for each parameter
      loc_loc_priors:           location of the prior for the mean of each
                                parameter
      loc_scale_priors:         scale of the prior of the mean of each
                                parameter
      scale_scale_priors:       scale of the prior of the standard
                                deviation of each parameter
      noise_priors:             prior on the white noise
      loc_suffix (optional):    append to the variable name to give the top
                                level sample for the location
      scale_suffix (optional):  append to the variable name to give the top
                                level sample for the scale
      include_noise (optional): if true include white noise in the inference
  """
  def __init__(self, maker, names, loc_loc_priors, loc_scale_priors,
      scale_scale_priors, noise_prior, loc_suffix = "_loc",
      scale_suffix = "_scale", param_suffix = "_param",
      include_noise = False):
    super().__init__()
    
    # Store things we might later 
    self.maker = maker
    self.loc_suffix = loc_suffix
    self.scale_suffix = scale_suffix
    self.param_suffix = param_suffix
    self.include_noise = include_noise

    self.names = names

    # We need these for the shapes...
    self.loc_loc_priors = loc_loc_priors
    self.loc_scale_priors = loc_scale_priors
    self.scale_scale_priors = scale_scale_priors
    self.noise_prior = noise_prior

    # What type of noise are we using
    self.type_noise = noise_prior.dim() > 0

    # Setup both the top and bottom level variables
    self.bot_vars = names
    self.top_vars = []
    self.dims = []
    for var, loc_loc, loc_scale, scale_scale, in zip(
        names, loc_loc_priors, loc_scale_priors, scale_scale_priors):
      # These set standard PyroSamples with names of var + suffix
      dim = loc_loc.dim()
      self.dims.append(dim)
      self.top_vars.append(var + loc_suffix)
      setattr(self, self.top_vars[-1], PyroSample(dist.Normal(loc_loc,
        loc_scale).to_event(dim)))
      self.top_vars.append(var + scale_suffix)
      setattr(self, self.top_vars[-1], PyroSample(dist.HalfNormal(scale_scale
        ).to_event(dim)))
      
      # The tricks are: 1) use lambda self and 2) remember how python binds...
      setattr(self, var, PyroSample(
        lambda self, var = var, dim = loc_loc.dim(): dist.Normal(
          getattr(self, var + loc_suffix),
          getattr(self, var + scale_suffix)).to_event(dim)))

    # Setup the noise
    if self.include_noise:
      if self.type_noise:
        self.eps = PyroSample(dist.HalfNormal(noise_prior).to_event(1))
      else:
        self.eps = PyroSample(dist.HalfNormal(noise_prior))
    else:
      self.eps = noise_prior

    # This annoyance is required to make the adjoint solver work
    self.extra_param_names = []

  @property
  def nparams(self):
    return len(self.names)

  def sample_top(self):
    """
      Sample the top level variables
    """
    return [getattr(self, name) for name in self.top_vars]

  def sample_bot(self):
    """
      Sample the bottom level variables
    """
    return [getattr(self, name) for name in self.bot_vars]

  def make_guide(self):
    """
      Make the guide and cache the extra parameter names the adjoint solver
      is going to need
    """
    def guide(exp_data, exp_cycles, exp_types, exp_control, exp_results = None):
      # Setup and sample the top-level loc and scale
      top_loc_samples = []
      top_scale_samples = []
      for var, loc_loc, loc_scale, scale_scale, in zip(
          self.names, self.loc_loc_priors, self.loc_scale_priors, 
          self.scale_scale_priors):
        dim = loc_loc.dim()
        loc_param = pyro.param(var + self.loc_suffix + self.param_suffix, loc_loc)
        scale_param = pyro.param(var + self.scale_suffix + self.param_suffix, scale_scale,
            constraint = constraints.positive)
        
        top_loc_samples.append(pyro.sample(var + self.loc_suffix, dist.Delta(loc_param).to_event(dim)))
        top_scale_samples.append(pyro.sample(var + self.scale_suffix, dist.Delta(scale_param).to_event(dim)))

      # Add in the noise, if included in the inference
      if self.include_noise:
        eps_param = pyro.param("eps" + self.param_suffix, torch.tensor(self.noise_prior), constraint = constraints.positive)
        if self.type_noise:
          eps_sample = pyro.sample("eps", dist.Delta(eps_param).to_event(1))
        else:
          eps_sample = pyro.sample("eps", dist.Delta(eps_param))

      # Plate on experiments and sample individual values
      with pyro.plate("trials", exp_data.shape[2]):
        for i,(name, val, dim) in enumerate(zip(self.names, self.loc_loc_priors, self.dims)):
          # Fix this to init to the mean (or a sample I guess)
          ll_param = pyro.param(name + self.param_suffix, torch.zeros_like(val).unsqueeze(0).repeat((exp_data.shape[2],) + (1,)*dim) + 0.5)
          param_value = pyro.sample(name, dist.Delta(ll_param).to_event(dim))
    
    self.extra_param_names = [var + self.param_suffix for var in self.names]

    return guide

  def get_extra_params(self):
    """
      Actually list the extra parameters required for the adjoint solve.

      We can't determine this by introspection on the base model, so
      it needs to be done here
    """
    # Do some consistency checking
    for p in self.extra_param_names:
      if p not in pyro.get_param_store().keys():
        raise ValueError("Internal error, parameter %s not in store!" % p)

    return [pyro.param(name).unconstrained() for name in self.extra_param_names]

  def forward(self, exp_data, exp_cycles, exp_types, exp_control, exp_results = None):
    """
      Evaluate the forward model, optionally conditioned by the experimental
      data.

      Args:
        exp_data:       input data, (3,ntime,nexp)
        exp_cycles:     cycle labels (ntime,nexp)
        exp_types:      experiment types (nexp,)
        exp_control:    strain/stress control labels (nexp,)

      Additional Args:
        exp_results:    true results for conditioning (ntime,nexp)
    """
    # Sample the top level parameters
    curr = self.sample_top()
    eps = self.eps
    
    # Setup the full noise, which can be type specific
    if self.type_noise:
      full_noise = torch.empty(exp_data.shape[-1], device = exp_data.device)
      for i in experiments.exp_map.values():
        full_noise[exp_types==i] = eps[i]
    else:
      full_noise = eps

    with pyro.plate("trials", exp_data.shape[2]):
      # Sample the bottom level parameters
      bmodel = self.maker(*self.sample_bot(),
          extra_params = self.get_extra_params())
      # Generate the results
      predictions = bmodel.solve_both(exp_data[0], exp_data[1], exp_data[2], exp_control)
      # Process the results
      results = experiments.convert_results(predictions[:,:,0], exp_cycles, exp_types)

      # Sample!
      with pyro.plate("time", exp_data.shape[1]):
        pyro.sample("obs", dist.Normal(results, full_noise), obs = exp_results)

    return results
