"""
  Various routines for dealing with generating input and processing output
  for abstracted records of experimental tests.
  
  These routines do things like:
    * Convert simplified descriptions of tests into strain/strain time
      histories that the models can use
    * Generate random experiments and the corresponding inputs

  The actual deterministic and statistical fitting routines expect data
  in a common format.  There are a few conditions on the input test data:

  1. You have to sample all tests with the same number of time points (ntime).
  2. You have to provide the full, correct history of the input conditions.
     For strain controlled experiments this is the full history of test
     time, strain, and temperature.  For stress control it is the full
     history of time, stress, and temperature.
  3. You do not have to provide the full *results* history.  That is,
     the fit routines can accommodate some types of abstracted test data,
     like maximum stress as a function of cycles for strain-controlled cyclic
     tests.  The convert_results function can then convert a full (simulated)
     results history to the right abstracted format for comparison to the
     test data.

  The fitting routines rely on five tensors with the following format:

  * data (3, ntime, nexp): the input experimental conditions.  The first
    index contains the experimental time, temperature, and strain history
    for strain controlled tests and the experimental time, temperature
    and stress history for stress controlled tests
  * results (ntime, nexp): the (potentially abstracted) experimental results.
    For example, this could be the stress history for a tensile test or
    the strain history for a creep test.  The specific requirements for
    certain tests types are described below.
  * cycles (ntime, nexp): the cycle counts for cyclic  tests or an integer
    used to indicate features of the experiment for other test types.
    Specific requirements are given below.
  * types (nexp,): an integer giving the test type (see list below)
  * control (nexp,): 0 for strain control, 1 for stress control

  The current test types are:
  0: "tension" -- uniaxial, monotonic tension or compression test
  1: "relaxation" -- stress relaxation test
  2: "strain_cyclic" -- strain controlled cyclic (i.e. creep or creep-fatigue)
  3: "creep" -- a creep test
  4: "stress_cyclic" -- a stress controlled cyclic test
  5: "abstract_tensile" -- tension tests where only the yield strength
     and ultimate tensile strength are known
  6: "direct_data" -- the full results history is known and provided

  The load_results function provides a way to load data for this type
  from an xarray format.  The same tensors are stored in the xarray 
  structure, however the descriptive string names for the tests are used
  and "strain" or "stress" is used instead of 0 and 1 to indicate the
  control type.  Additional metadata can be stored in the xarray format with
  source information, heat ID, etc.
"""

import numpy as np
import numpy.random as ra
import itertools

import torch

def load_results(xdata, device = torch.device("cpu")):
  """
    Load experimental data from xarray into torch tensors

    Args:
      xdata:    xarray data structure

    Additional Args:
      device:   the device to dump the resulting arrays on

    The output format is critical, this function returns 5 tensors:

      data (3, ntime, nexperiment)

        The initial index are the experimental (times, temperatures, idata)
        For strain controlled tests idata is strain
        For stress controlled tests idata is stress

      results (ntime, nexperiment)

        For strain controlled tests this is stress
        For stress controlled tests this is strain

      cycles (ntime, nexperiment)

        Cycle count for all tests

      types (nexperiment,)

        Experiment types, converted to integers per the dicts above

      control (nexperiment,)

        Maps the string control type ("strain" or "stress") to an integer
        using the dict above
  """
  time = torch.tensor(xdata["time"].values, device = device)
  temp = torch.tensor(xdata["temperature"].values, device = device)
  strain = torch.tensor(xdata["strain"].values, device = device)
  stress = torch.tensor(xdata["stress"].values, device = device)

  cycle = torch.tensor(xdata["cycle"].values, device = device)
  types = torch.tensor([exp_map[t] for t in xdata["type"].values], device = device)
  control = torch.tensor([control_map[t] for t in xdata["control"].values], device = device)

  data = torch.empty((4,) + time.shape, device = device)

  data[0] = time
  data[1] = temp
  data[2,:,control==0] = strain[:,control==0]
  data[2,:,control==1] = stress[:,control==1]

  results = torch.empty_like(time)
  results[:,control==0] = stress[:,control==0]
  results[:,control==1] = strain[:,control==1]

  return data, results, cycle, types, control

def convert_results(results, cycles, types):
  """
    Process a raw results vector to our common format based on test type

    Args:
      results:      raw results data (ntime, nexperiment)
      cycles:       cycle counts (ntime, nexperiment)
      types:        test types (nexperiment,)
  """
  processed = torch.empty_like(results)

  for i, func in exp_fns_num.items():
    current = types == i
    if torch.sum(current) > 0:
      processed[:,current] = func(cycles[:,current], results[:,current])

  return processed

def format_direct_data(cycles, predictions):
  """
    Format direct stress/strain results

    Args:
      cycles:       cycle count
      predictions:  input to format
    
    For this test type cycles just equals 0 for all time steps

    Do nothing, just return the predictions.
  """
  return predictions

def format_abstract_tensile(cycles, predictions):
  """
    Format abstracted tensile test data, where only the yield strength
    and ultimate tensile strength are available.

    This method relies on the input cycles being:
    * 0: in the elastic regime
    * 1: for the first 1/2 of the remaining time points outside of the elastic
         regime
    * 2: for the second 1/2 of the remaining time points.

    The inputs are the full stress history.  This function:
    * Cycle 0: overwrite with zeros
    * Cycle 1: overwrite with the first value in cycle 1
    * Cycle 2: overwrite with the maximum stress

    Args:
      cycles:       cycle count
      predictions:  input to format
  """
  result = torch.zeros_like(predictions)
  
  # cycles == 0 -> 0
  result += torch.where(cycles==0, 0.0, 0.0)

  # cycles == 1 -> first value in that region
  _, index = torch.max(cycles==1, 0)
  result += torch.where(cycles==1, 
      predictions.gather(0, index.view(1,-1))[0], 0.0)

  # cycles == 2 -> max value overall
  mvalues, _ = torch.max(predictions, 0)
  result += torch.where(cycles==2, mvalues, 0.0)

  return result

def format_tensile(cycles, predictions):
  """
    Format tension test data to our "post-processed" form for comparison

    Input data are stresses for this test type

    Cycles are all 0

    Do nothing!

    Args:
      cycles:       cycle count/listing
      predictions:  input data
  """
  return predictions

def format_relaxation(cycles, predictions):
  """
    Format stress relaxation test data to our "post-processed" form for 
    comparison. This works for both creep and stress relaxation tests.

    Input data are stresses for stress relaxation and strains for creep

    Cycle 0 indicates the loading part of the test, cycle 1 is the hold

    Zero out the loading results, replace the relaxation results with the
    normalized (subtract t=0) curve

    Args:
      cycles:       cycle count/listing
      predictions:  input data
  """
  result = torch.zeros_like(predictions)
  rcurve = cycles[:,0] == 1 # This is right, but dangerous in the future

  curve = predictions[rcurve]
  curve = curve - curve[0]

  result[rcurve] = curve

  return result

def format_cyclic(cycles, predictions):
  """
    Format a generic cyclic test -- works for both stress and strain control

    Args:
      cycles:       cycle count/listing
      predictions:  input data

    Input data are stresses for strain control and strains for stress control.

    We format this as a "block" -- the values for each cycle are replaced
    by the maximum value within the cycle
  """
  # If this is slow we can probably remove the for loop
  result = torch.zeros_like(predictions)
  uc = cycles[:,0] # Correct but dangerous for future expansion
  for i in range(uc[-1]+1):
    curr = uc == i
    vals, _ = torch.max(predictions[curr], axis = 0)
    result[curr] = vals
  
  return result

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

# Numerical codes for each test type
exp_map = {"tensile": 0, "relaxation": 1, "strain_cyclic": 2, 
    "creep": 3, "stress_cyclic": 4, "abstract_tensile": 5, "direct_data": 6}
# Function to use to process each test type
exp_fns = {"tensile": format_tensile, "relaxation": format_relaxation,
    "strain_cyclic": format_cyclic, "creep": format_relaxation,
    "stress_cyclic": format_cyclic, "abstract_tensile": format_abstract_tensile,
    "direct_data": format_direct_data}
# Map to numbers instead
exp_fns_num = {exp_map[k]: v for k,v in exp_fns.items()}

# Map control type to number
control_map = {"strain": 0, "stress": 1}
