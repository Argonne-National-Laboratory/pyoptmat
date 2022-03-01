"""
  A few common functions related to solving nonlinear and linear systems
  of equations.  These functions exist to "torchify" these common tasks.
  As always, the routines accept batched input.
"""

import torch
import warnings

def newton_raphson(fn, x0, linsolver = "lu", rtol = 1e-6, atol = 1e-10,
    miter = 100):
  """
    Solve a nonlinear system with Newton's method.  Return the
    solution and the last Jacobian

    Args:
      fn (function):        function that returns the residual and Jacobian
      x0 (torch.tensor):    starting point

    Keyword Args:
      linsolver (string):   method to use to solve the linear system, options are
                            "diag" or "lu".  Defaults to "lu".  See 
                            :py:func:`pyoptmat.solvers.solve_linear_system`
      rtol (float):         nonlinear relative tolerance
      atol (float):         nonlinear absolute tolerance
      miter (int):          maximum number of nonlinear iterations

    Returns:
      torch.tensor, torch.tensor:   solution to system of equations and Jacobian evaluated at that point
  """
  x = x0
  R, J = fn(x)

  nR = torch.norm(R, dim = -1)
  nR0 = nR
  i = 0

  while (i < miter) and torch.any(nR > atol) and torch.any(nR / nR0 > rtol):
    x -= solve_linear_system(J, R)
    R, J = fn(x)
    nR = torch.norm(R, dim = -1)
    i += 1
 
  if i == miter:
    warnings.warn("Implicit solve did not succeed.  Results may be inaccurate...")
  
  return x, J

def solve_linear_system(A, b, method = "lu"):
  """
    Solve or iterate on a linear system of equations using one of
    several methods:

    * "lu" -- use :code:`torch.linalg.solve`
    * "diag" -- do one Jacobi iteration (:code:`b / diag(A)`)

    Args:
      A (torch.tensor):     block matrix
      b (torch.tensor):     block RHS
    
    Keyword Args:
      method (string):      Method to use.  Options outlined above.
  """
  if method == "diag":
    return b / torch.diagonal(A, dim1=-2, dim2=-1)
  elif method == "lu":
    return torch.linalg.solve(A, b)
  else:
    raise ValueError("Unknown solver method!")
