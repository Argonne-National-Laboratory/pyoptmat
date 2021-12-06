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

def construct_weights(etypes, weights, normalize = True):
  """
    Construct an array of weights 

    Args:
      etypes:               strings giving the experiment type
      weights:              dictionary mapping etype to weight
      normalize (optional): normalize by the number of experiments of each type
  """
  warray = torch.ones(len(etypes))

  count = defaultdict(int)
  for i,et in enumerate(etypes):
    warray[i] = weights[et]
    count[et] += 1

  if normalize:
    for i, et in enumerate(etypes):
      warray[i] /= count[et]

  return warray

def expand_indices(base, total, ntimes):
  l = base.shape[0]
  new = base.repeat(ntimes)
  for i in range(ntimes):
    new[i*l:(i+1)*l] += i*total

  return new

def grid_search(model, idata, loss, bounds, 
    ngrid, method = "lhs-maximin", save_grid = None,
    save_every = 1, rbf_function = "inverse", nan_v = 1e30,
    nbatch = 1):
  """
    Use a coarse grid search to find a good starting point

    Args:
      model:                    forward model
      loss:                     loss function
      bounds:                   bounds on each parameter
      ngrid:                    number of points
      method (optional):        method for generating the grid
      save_grid (optional):     save the parameter grid to a file for future use
      rbf_function (optional):  kernel for radial basis interpolation
  """
  # Extract the data from the tuple
  idata, cycles, types, inds = idata
  ntotal = idata.shape[-1]

  # Parameter order
  params = [(n, p.shape, torch.flatten(p).shape[0]) for n, p
      in list(model.named_parameters())]
  offsets = [0] + list(np.cumsum([i[2] for i in params]))

  device = idata.device
  
  # Generate the space
  # Need to do this on the cpu
  space = []
  for n, shp, sz in params:
    for i in range(sz):
      space.append((bounds[n][0].cpu().numpy().flatten()[i],
        bounds[n][1].cpu().numpy().flatten()[i]))
  sspace = Space(space)

  # Actually sample
  if method == "lhs-maximin":
    sampler = Lhs(criterion = "maximin")
  else:
    raise ValueError("Unknown sampling method %s" % method)
  
  samples = torch.tensor(sampler.generate(sspace.dimensions, ngrid),
      device = device)

  data = torch.zeros((samples.shape[0], samples.shape[1]+1), device = device)
  
  exp_results = experiments.setup_experiment_vector(idata[:,:,inds[0]], idata[:,:,inds[1]])
  locs = torch.mean(exp_results, 0)
  #scales = torch.std(exp_results, 0)
  scales = torch.abs(locs)

  # Divide into blocks
  pblocks = torch.split(samples, nbatch)
  pindices = torch.split(torch.arange(samples.shape[0]), nbatch)
  
  # Here we go
  t = tqdm(enumerate(zip(pblocks, pindices)), total = len(pblocks), 
      desc = "Grid sampling")
  for i,(csample,indices) in t:
    ncurr = csample.shape[0] 

    for k,(name, shp, sz) in enumerate(params):
      getattr(model,name).data = torch.repeat_interleave(csample[:,offsets[k]:offsets[k+1]].reshape((ncurr,) + shp), ntotal, 0)
    
    with torch.no_grad():
      res = model(idata.repeat(1,1,ncurr),
          cycles.repeat(1,ncurr),
          types.repeat(ncurr),
          (expand_indices(inds[0], ntotal, ncurr), expand_indices(inds[1], ntotal, ncurr)))
      
      res = res.reshape(res.shape[0], ncurr, ntotal)
      
      for i,(pv, ind) in enumerate(zip(csample, indices)):
        data[ind,:-1] = csample[i]
        lv = loss((res[:,i]-locs)/scales, (exp_results-locs)/scales)
        data[ind,-1] = torch.nan_to_num(lv, nan = nan_v)

    # Store results if requested
    if save_grid is not None and i % save_every == 0:
      torch.save(data, save_grid)
  
  # We now need to do this on the CPU again
  data = data.cpu().numpy()
  
  # Toss Nans
  keep = data[:,-1] < nan_v
  data = data[keep]
  
  # Scale data
  mean = np.mean(data, axis = 0)
  std = np.std(data, axis = 0)
  scaled_input = (data[:,:-1] - mean[:-1]) / std[:-1]
  # Log fn seems to work best
  obj = np.log(data[:,-1])
  # Scale bounds
  scaled_bounds = [sorted([(b[0]-m)/s,(b[1]-m)/s]) for b,m,s in zip(space, mean, std)]
  
  # Get the surrogate and minimize it
  ifn = inter.RBFInterpolator(scaled_input, obj)
  res = opt.minimize(lambda x: ifn(x[None,:]), np.zeros((len(mean)-1,)),
      method = 'L-BFGS-B', bounds = scaled_bounds)
  if not res.success:
    warnings.warn("Surrogate model minimization did not succeed!")

  x = np.array([xi*s+m for xi,s,m in zip(res.x, mean, std)])

  # Setup the parameter dict and alter the model
  result = {}
  for k, (name, shp, sz) in enumerate(params):
    getattr(model, name).data = torch.tensor(x[offsets[k]:offsets[k+1]]).reshape(shp).to(device)
    result[name] = x[offsets[k]:offsets[k+1]].reshape(shp)
  
  return result

def bounded_scale_function(bounds):
  """
    Sets up a scaling function that maps `(0,1)` to `(bounds[0], bounds[1])`
    and clips the values to remain in that range

    Args:
      bounds:   tuple giving the parameter bounds
  """
  return lambda x: torch.clamp(x, 0, 1)*(bounds[1]-bounds[0]) + bounds[0]

def clamp_scale_function(bounds):
  """
    Just clamps

    Args:
      bounds:   tuple giving the parameter bounds
  """
  return lambda x: torch.clamp(x, bounds[0], bounds[1])

class DeterministicModelExperiment(Module):
  """
    Wrap a material model to provide a :py:mod:`pytorch` deterministic model

    Args:
      maker:      function that returns a valid Module, given the 
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

  def forward(self, exp_data, exp_cycles, exp_types, exp_inds):
    """
      Integrate forward and return the results

      Args:

    """
    model = self.maker(*self.get_params())

    predictions = model.solve_both(exp_data[0], exp_data[2],
        exp_data[1], exp_inds)
  
    return experiments.assemble_results_full(exp_data, exp_cycles, exp_types, predictions[:,:,0])

class DeterministicModel(Module):
  """
    Wrap a material model to provide a :py:mod:`pytorch` deterministic model

    Args:
      maker:      function that returns a valid Module, given the 
                  input parameters
      names:      names to use for the parameters
      ics:        initial conditions to use for each parameter
  """
  def __init__(self, maker, names, ics, mode = "strain"):
    super().__init__()

    self.maker = maker
    
    # Add all the parameters
    self.params = names
    for name, ic in zip(names, ics):
      setattr(self, name, Parameter(torch.tensor(ic)))

  def get_params(self):
    """
      Return the parameters for input to the model
    """
    return [getattr(self, name) for name in self.params]

  def forward(self, times, input_data, temperatures, mode = "strain"):
    """
      Integrate forward and return the results

      Args:
        times:          time points to hit
        input_data:     input data
        temperatures:   input temperature data
    """
    model = self.maker(*self.get_params())
    
    if mode == "strain":
      return model.solve_strain(times, input_data, temperatures)[:,:,0]
    else:
      return model.solve_stress(times, input_data, temperatures)[:,:,0]

class StatisticalModel(PyroModule):
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
      eps:        random noise, could be a constant value or a parameter
  """
  def __init__(self, maker, names, locs, scales, eps):
    super().__init__()

    self.maker = maker
    
    # Add all the parameters
    self.params = names
    for name, loc, scale in zip(names, locs, scales):
      setattr(self, name, PyroSample(dist.Normal(loc, scale)))

    self.eps = eps

  def get_params(self):
    """
      Return the sampled parameters for input to the model
    """
    return [getattr(self, name) for name in self.params]

  def forward(self, times, strains, temperatures, true = None):
    """
      Integrate forward and return the result

      Args:
        times:          time points to hit in integration
        strains:        input strains
        temperatures:   input temperatures

      Additional Args:
        true:       true values of the stress, if we're using this
                    model in inference
    """
    model = self.maker(*self.get_params())
    stresses = model.solve_strain(times, strains, temperatures)[:,:,0]
    with pyro.plate("trials", times.shape[1]):
      with pyro.plate("time", times.shape[0]):
        return pyro.sample("obs", dist.Normal(stresses, self.eps), obs = true)

class HierarchicalStatisticalModel(PyroModule):
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
      self.eps = PyroSample(dist.HalfNormal(noise_prior))
    else:
      self.eps = torch.tensor(noise_prior)

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
    def guide(times, strains, temperatures, true_stresses = None):
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
        eps_sample = pyro.sample("eps", dist.Delta(eps_param))

      # Plate on experiments and sample individual values
      with pyro.plate("trials", times.shape[1]):
        for i,(name, val, dim) in enumerate(zip(self.names, self.loc_loc_priors, self.dims)):
          ll_param = pyro.param(name + self.param_suffix, torch.zeros_like(val).unsqueeze(0).repeat((times.shape[1],) + (1,)*dim))
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

  def forward(self, times, strains, temperatures, true_stresses = None):
    """
      Evaluate the forward model, conditioned by the true stresses if
      they are available

      Args:
        times:                      time points to hit
        strains:                    input strains
        temperaturse:               input temperature data
        true_stresses (optional):   actual stress values, if we're conditioning
                                    for inference
    """
    # Sample the top level parameters
    curr = self.sample_top()
    eps = self.eps

    with pyro.plate("trials", times.shape[1]):
      # Sample the bottom level parameters
      bmodel = self.maker(*self.sample_bot(),
          extra_params = self.get_extra_params())
      # Generate the stresses
      stresses = bmodel.solve_strain(times, strains, temperatures)[:,:,0]
      # Sample!
      with pyro.plate("time", times.shape[0]):
        pyro.sample("obs", dist.Normal(stresses, eps), obs = true_stresses)

    return stresses
