"""
  Various routines for dealing with generating input and processing output
  for abstracted records of experimental tests.
  
  These routines do things like:
    * Convert simplified descriptions of tests into strain/strain time
      histories that the models can use
    * Generate random experiments and the corresponding inputs

  We define the various tests with a dictionary

  Strain controlled load cycles are defined by a dictionary with:
    * "max_strain" -- the maximum strain 
    * "R" -- the ration between min_strain/max_strain
    * "strain_rate" -- the loading rate
    * "tension_hold" -- tensile hold time
    * "compression_hold" -- compression hold time

  Tension tests are defined by:
    * "strain_rate" -- loading rate
    * "max_strain" -- maximum strain
"""

import numpy as np
import numpy.random as ra
import itertools

import torch

exp_map_strain = {"tensile": 0, "relaxation": 1, "strain_cyclic": 3}
exp_map_stress = {"creep": 4, "stress_cyclic": 5}
exp_map = {v:k for k,v in itertools.chain(exp_map_strain.items(), 
  exp_map_stress.items())}

def setup_experiment_vector(strain_data, stress_data, stress_scale = 100.0):

  ntime = strain_data.shape[1]
  nstrain = strain_data.shape[2]
  nstress = stress_data.shape[2]
  ntotal = nstrain + nstress

  exp_result = torch.empty(ntime, ntotal)

  exp_result[:,:nstrain] = strain_data[3]
  exp_result[:,nstrain:] = stress_data[3] * stress_scale

  return exp_result

def assemble_results(strain_data, strain_cycles, strain_types, pred_strain,
    stress_data, stress_cycles, stress_types, pred_stress, stress_scale = 100.0):
  """

  """
  # Sizes
  ntime = strain_data.shape[1]
  nstrain = strain_data.shape[2]
  nstress = stress_data.shape[2]
  ntotal = nstrain + nstress

  model_result = torch.empty(ntime, ntotal)

  n = 0

  # Tensile tests
  tensile = strain_types == exp_map_strain["tensile"]
  ni = torch.sum(tensile)

  model_result[:,n:n+ni] = format_tensile(
      strain_cycles[:,tensile], pred_strain[:,tensile])
  n += ni

  # Relaxation tests
  relaxation = strain_types == exp_map_strain["relaxation"]
  ni = torch.sum(relaxation)

  model_result[:,n:n+ni] = format_relaxation(
      strain_cycles[:,relaxation], pred_strain[:,relaxation])
  n += ni

  # Strain-controlled cyclic
  ecyclic = strain_types == exp_map_strain["strain_cyclic"]
  ni = torch.sum(ecyclic)
  
  model_result[:,n:n+ni] = format_cyclic(
      strain_cycles[:,ecyclic], pred_strain[:,ecyclic])
  n += ni

  # Creep
  creep = stress_types == exp_map_stress["creep"]
  ni = torch.sum(creep)
  
  model_result[:,n:n+ni] = format_relaxation(
      stress_cycles[:,creep], pred_stress[:,creep]) * stress_scale

  n += ni

  # Stress controlled cyclic
  scyclic = stress_types == exp_map_stress["stress_cyclic"]
  ni = torch.sum(scyclic)

  model_result[:,n:n+ni] = format_cyclic(
      stress_cycles[:,scyclic], pred_stress[:,scyclic]) * stress_scale
  n += ni

  return model_result

def format_tensile(cycles, predictions):
  """
    Do nothing!
  """
  return predictions

def format_relaxation(cycles, predictions):
  """
    Zero out the loading results, replace the relaxation results with the
    normalized (subtract t=0) curve
  """
  result = torch.zeros_like(predictions)
  rcurve = cycles[:,0] == 1 # This is right, but dangerous in the future

  curve = predictions[rcurve]
  curve = curve - curve[0]

  result[rcurve] = curve

  return result

def format_cyclic(cycles, predictions):
  # If this is slow we can probably remove the for loop
  result = torch.zeros_like(predictions)
  uc = cycles[:,0] # Correct but dangerous for future expansion
  for i in range(uc[-1]+1):
    curr = uc == i
    vals, _ = torch.max(predictions[curr], axis = 0)
    result[curr] = vals
  
  return result

def format_strain_control(ds):
  data = torch.stack((torch.tensor(ds["time"].values), 
    torch.tensor(ds["strain"].values), 
    torch.tensor(ds["temperature"].values),
    torch.tensor(ds["stress"].values)))
  cycle = torch.tensor(ds["cycle"].values)
  
  types = torch.tensor([exp_map_strain[t] for t in ds["type"].values])
  
  return data, cycle, types

def format_stress_control(ds):
  data = torch.stack((torch.tensor(ds["time"].values), 
    torch.tensor(ds["stress"].values), 
    torch.tensor(ds["temperature"].values),
    torch.tensor(ds["strain"].values)))
  cycle = torch.tensor(ds["cycle"].values)
  
  types = torch.tensor([exp_map_stress[t] for t in ds["type"].values])
  
  return data, cycle, types

def make_tension_tests(rates, temperatures, elimits, nsteps):
  """
    Produce tension test (time,strain,temperature) history blocks
    given tensor inputs for the strain rates, temperatures, and
    maximum strain of each test

    Args:
      rates:            1D tensor giving the strain rate of each test
      temperaturess:    1D tensor giving the constant temperature of each test
      elimits:          1D tensor giving the maximum strain of each test
      nsteps:           integer number of steps
  """
  nbatch = temperatures.shape[0]
  times = torch.zeros(nsteps, nbatch)
  strains = torch.zeros_like(times)
  temps = torch.zeros_like(strains)
  
  for i in range(nbatch):
    times[:,i] = torch.linspace(0, elimits[i]/rates[i], nsteps)
    strains[:,i] = torch.linspace(0, elimits[i], nsteps)
    temps[:,i] = temperatures[i]

  return times, strains, temps, torch.zeros_like(times, dtype = int)

def make_creep_tests(stress, temperature, rate, hold_times,
    nsteps_load, nsteps_hold, logspace = False):
  """
    Produce creep test input (time,stress,temperature) given tensor
    inputs for the target stress, target temperature, loading rate

    Args:
      stress:               1D tensor of target stresses
      temperature:          1D tensor of target temperature
      rate:                 1D tensor of target rates
      hold_times:           1D tensor of hold times
      nsteps_load:          number of time steps to load up the sample
      nsteps_hold:          number of time steps to hold the sample
      logspace (optional):  log space the hold time steps
  """
  nbatch = stress.shape[0]
  nsteps = nsteps_load + nsteps_hold
  
  stresses = torch.zeros(nsteps, nbatch)
  times = torch.zeros_like(stresses)
  temperatures = torch.zeros_like(stresses)
  
  for i, (s,t,lr,T) in enumerate(zip(stress,hold_times,rate,temperature)):
    stresses[:nsteps_load,i] = torch.linspace(0, s, nsteps_load)
    stresses[nsteps_load:,i] = s

    times[:nsteps_load,i] = torch.linspace(0, s / lr, nsteps_load)
    temperatures[:,i] = T
    if logspace:
      times[nsteps_load:,i] = torch.logspace(torch.log10(times[nsteps_load-1,i]),
          torch.log10(t), nsteps_hold+1)[1:]
    else:
      times[nsteps_load:,i] = torch.linspace(times[nsteps_load-1,i], t, nsteps_hold+1)[1:]
  
  cycles = torch.ones_like(times, dtype = int)
  cycles[:nsteps_load] = 0

  return times, stresses, temperatures, cycles

def generate_random_tension(strain_rate = [1.0e-6,1.0e-2], 
    max_strain = 0.2):
  """
  Generate a random tension test condition in the provided ranges

  Args:
    strain_rate (optional): Range of strain rates
    max_strain (optional):  Maximum strain to simulate
  """
  return {
      "max_strain": max_strain,
      "strain_rate": 10.0**ra.uniform(np.log10(strain_rate[0]),
        np.log10(strain_rate[1]))
      }

def sample_tension(test, nsteps = 50):
  """Generate the times and strains for a tensile test

  Args:
    test:               Dictionary defining the test case
    nsteps (optional):  Number of steps to sample
  """
  tmax = test['max_strain'] / test['strain_rate']
  times = np.linspace(0,tmax,nsteps)
  strains = times * test['strain_rate']

  return times, strains

def generate_random_cycle(max_strain = [0,0.02], R = [-1, 1], 
    strain_rate = [1.0e-3, 1.0e-5], tension_hold = [0,1*3600.0],
    compression_hold = [0,600]):
  """
    Generate a random cycle in the provided ranges

    Args:
      max_strain:       range of the maximum strains
      R:                range of R ratios
      strain_rate:      range of loading strain rates
      tension_hold:     range of tension hold times
      compression_hold: range of compression hold times
  """
  return {
      "max_strain": ra.uniform(*max_strain),
      "R": ra.uniform(*R),
      "strain_rate": 10.0**ra.uniform(np.log10(strain_rate[0]), np.log10(strain_rate[1])),
      "tension_hold": ra.uniform(*tension_hold),
      "compression_hold": ra.uniform(*compression_hold)
      }

def sample_cycle_normalized_times(cycle, N, nload = 10, nhold = 10):
  """
    Take a random cycle dictionary and expand into discrete 
    times, strains samples where times are the actual, physical
    times, given over the fixed phases

      * :math:`0 \\rightarrow  t_{phase}` -- tension load
      * :math:`t_{phase} \\rightarrow 2 t_{phase}` -- tension hold
      * :math:`2 t_{phase} \\rightarrow 3 t_{phase}` --   unload
      * :math:`3 t_{phase} \\rightarrow 4 t_{phase}` -- compression load
      * :math:`4 t_{phase} \\rightarrow 5 t_{phase}` -- compression hold
      * :math:`5 t_{phase} \\rightarrow 6 t_{phase}` -- unload

    This pattern repeats for N cycles

    Args:
      cycle:            dictionary defining the load cycle
      N:                number of repeats to include in the history
      nload (optional): number of steps to use for the load time
      nhold (optional): number of steps to use for the hold time
  """
  emax = cycle['max_strain']
  emin = cycle['R'] * cycle['max_strain']
  erate = cycle['strain_rate']

  # Segments:
  t1 = np.abs(emax) / erate
  t2 = cycle['tension_hold']
  t3 = np.abs(emax-emin) / erate
  t4 = cycle['compression_hold']
  t5 = np.abs(emin) / erate
  divisions = [t1, t2, t3, t4, t5]
  timesteps = [nload, nhold, 2*nload, nhold, nload]
  cdivisions = np.cumsum(divisions)
  period = cdivisions[-1]

  Ntotal = nload * 4 + nhold * 2

  times = np.zeros((1+Ntotal*N,))
  cycles = np.zeros(times.shape, dtype = int)

  n = 1
  tc = 0
  for k in range(N):
    for ti,ni in zip(divisions, timesteps):
      times[n:n+ni] = np.linspace(tc, tc + ti, ni+1)[1:]
      cycles[n:n+ni] = k
      n += ni
      tc += ti
  
  tp = times % period
  strains = np.piecewise(tp,
      [
        np.logical_and(tp >= 0, tp < cdivisions[0]),
        np.logical_and(tp >= cdivisions[0], tp < cdivisions[1]),
        np.logical_and(tp >= cdivisions[1], tp < cdivisions[2]),
        np.logical_and(tp >= cdivisions[2], tp < cdivisions[3]), 
        np.logical_and(tp >= cdivisions[3], tp < cdivisions[4]),
      ],
      [
        lambda tt: tt / t1 * emax,
        lambda tt: tt * 0 + emax,
        lambda tt: emax - (tt - cdivisions[1]) / t3 * (emax-emin),
        lambda tt: tt * 0 + emin,
        lambda tt: emin - (tt - cdivisions[3]) / t5 * emin
      ])

  return times, strains, cycles

