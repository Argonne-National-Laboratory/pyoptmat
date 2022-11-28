"""
  A few common functions related to solving nonlinear and linear systems
  of equations.  These functions exist to "torchify" these common tasks.
  As always, the routines accept batched input.
"""

import warnings
import torch

class LUBackSolve(torch.autograd.Function):
    """
    This illustrates how to calculate the backward pass of A x = b
    using adjoints

    This is not currently used because we don't need the backward pass
    of a linear solve
    """
    @staticmethod
    def forward(ctx, A, b):
        LU, P = torch.linalg.lu_factor(A)
        x = torch.linalg.lu_solve(LU, P, b.unsqueeze(-1)).squeeze(-1)
        ctx.save_for_backward(x, LU, P)

        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, LU, P = ctx.saved_tensors

        f = torch.linalg.lu_solve(LU, P, grad_output.unsqueeze(-1), adjoint = True).squeeze(-1)
        z = -torch.einsum('bi,bj->bij', (f, x))

        return z, f

def newton_raphson(fn, x0, linsolver="lu", rtol=1e-6, atol=1e-10, miter=100):
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
      torch.tensor, torch.tensor:   solution to system of equations and
                                    Jacobian evaluated at that point
    """
    # Determine the solver function
    if x0.shape[-1] == 1:
        solver = diagonal_linear_solve
    elif linsolver == "lu":
        solver = lu_linear_solve
    elif linsolver == "diag":
        solver = diagonal_linear_solve
    elif linsolver == "jacobi":
        solver = jacobi_linear_solve
    else:
        raise ValueError(f"Unknown linear solver type {linsolver}")

    x = x0
    R, J = fn(x)

    nR = torch.norm(R, dim=-1)
    nR0 = nR
    i = 0

    while (i < miter) and torch.any(nR > atol) and torch.any(nR / nR0 > rtol):
        x -= solver(J, R)
        R, J = fn(x)
        nR = torch.norm(R, dim=-1)
        i += 1

    if i == miter:
        warnings.warn("Implicit solve did not succeed.  Results may be inaccurate...")

    return x, J

def lu_linear_solve(A, b):
    """
    Solve a linear system of equations with the built in
    torch.linalg.solve

    Args:
      A (torch.tensor):     block matrix
      b (torch.tensor):     block RHS
      x0 (torch.tensor):    guess on solution
    """
    return torch.linalg.solve(A, b)

def diagonal_linear_solve(A, b):
    """
    Do one iteration of the Jacobi method on the provided
    linear system of equations

    Args:
      A (torch.tensor):     block matrix
      b (torch.tensor):     block RHS
      x0 (torch.tensor):    guess on solution
    """
    return b / torch.diagonal(A, dim1=-2, dim2=-1)

def jacobi_linear_solve(A, b, atol : float = 1e-12, miter : int = 25):
    """
    Solve a system using the Jacobi method

    Args:
      A (torch.tensor):     block matrix
      b (torch.tensor):     block RHS
      x0 (torch.tensor):    guess on solution
      atol (float):         nonlinear absolute tolerance
      miter (int):          maximum number of nonlinear iterations
    """
    x = torch.zeros_like(b)
    r = A.bmm(x.unsqueeze(-1)).squeeze(-1) - b
    nr = torch.norm(r, dim = -1)

    D = torch.diagonal(A, dim1=-2, dim2=-1)
    L = A - torch.diag_embed(D)

    i = 0

    while torch.any(nr > atol) and (i < miter):
        x = (b - L.bmm(x.unsqueeze(-1)).squeeze(-1)) / D
        r = A.bmm(x.unsqueeze(-1)).squeeze(-1) - b
        nr = torch.norm(r, dim = -1)
        i += 1

    return x
