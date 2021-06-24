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

import torch

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

  n = 1
  tc = 0
  for k in range(N):
    for ti,ni in zip(divisions, timesteps):
      times[n:n+ni] = np.linspace(tc, tc + ti, ni+1)[1:]
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

  return times, strains

