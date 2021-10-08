import torch
import warnings

def newton_raphson(fn, x0, linsolver = "lu", rtol = 1e-6, atol = 1e-10,
    miter = 100):
  """
    Solve a nonlinear system with Newton's method.  Return the
    solution and the last Jacobian

    Args:
      fn:                   function that returns the residual and Jacobian
      x0:                   starting point
      linsolver (optional): method to use to solve the linear system
      rtol (optional):      nonlinear relative tolerance
      atol (optional):      nonlinear absolute tolerance
      miter (optional):     maximum number of nonlinear iterations
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
    Solve or iterate on a linear system of equations

    Args:
      A:                    block matrix
      b:                    block RHS
      method (optional): 
  """
  if method == "diag":
    return b / torch.diagonal(A, dim1=-2, dim2=-1)
  elif method == "lu":
    return torch.linalg.solve(A, b)
  else:
    raise ValueError("Unknown solver method!")
