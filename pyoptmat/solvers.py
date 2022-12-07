"""
  A few common functions related to solving nonlinear and linear systems
  of equations.  These functions exist to "torchify" these common tasks.
  As always, the routines accept batched input.
"""

import warnings
import torch


def newton_raphson_bt(fn, x0, linsolver="lu", rtol=1e-6, atol=1e-10, miter=100,
        max_bt = 5):
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
      max_bt (int):         maximum number of backtracking line search iterations

    Returns:
      torch.tensor, torch.tensor:   solution to system of equations and
                                    Jacobian evaluated at that point
    """
    # Determine the solver function
    if x0.shape[-1] == 1:
        solver = jacobi_iteration_linear_solve
    elif linsolver == "lu":
        solver = lu_linear_solve
    elif linsolver == "diag":
        solver = jacobi_iteration_linear_solve
    else:
        raise ValueError(f"Unknown linear solver type {linsolver}")

    x = x0
    R, J = fn(x)

    nR = torch.norm(R, dim=-1)
    nR0 = nR
    i = 0
    
    alpha = torch.ones_like(x0)
    while (i < miter) and torch.any(nR > atol) and torch.any(nR / nR0 > rtol):
        dx = solver(J, R)
        nR_prev = nR.detach().clone()
        nR = nR_prev
        k = 0
        alpha = torch.ones_like(x0)
        while True:
            xp = x - alpha * dx
            R, J = fn(xp)
            nR = torch.norm(R, dim=-1)
            
            if torch.all(nR_prev > nR) or k > max_bt:
                break
            alpha[(nR - nR_prev) > atol] /= 2.0
            k += 1
        
        x = xp
        i += 1

    if i == miter:
        warnings.warn("Implicit solve did not succeed.  Results may be inaccurate...")

    return x, J

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
        solver = jacobi_iteration_linear_solve
    elif linsolver == "lu":
        solver = lu_linear_solve
    elif linsolver == "diag":
        solver = jacobi_iteration_linear_solve
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
    """
    return torch.linalg.solve_ex(A, b)[0]


def jacobi_iteration_linear_solve(A, b):
    """
    Do one iteration of the Jacobi method on the provided
    linear system of equations

    Args:
      A (torch.tensor):     block matrix
      b (torch.tensor):     block RHS
    """
    return b / torch.diagonal(A, dim1=-2, dim2=-1)

class NoOp():
    def __init__(self):
        pass

    def dot(self, x):
        return x

def gmres(A, b, x0 = None, tol = 1e-10, maxiter = 100, check = 5):
    """
    Solve a linear system of equations using GMRES

    Args:
        A (torch.tensor):   black matrix
        b (torch.tensor):   black RHS

    Keyword Args:
        x0 (torch.tensor):  initial guess, defaults to zeros
        tol (float):        absolute tolerance for solve
        maxiter (int):      maximum number of iterations
        check (int):        how often to check the residual

    Returns:
        x (torch.tensor):   block results
    """
    nbatch = A.shape[0]

    # Initial guess
    if not x0:
        x0 = torch.zeros_like(b)
    x = x0

    # Initial residual
    r = b - A.bmm(x.unsqueeze(-1)).squeeze(-1)

    # Basis
    V = torch.zeros((nbatch, b.shape[-1], maxiter+1), device = A.device)
    V[...,:,0] = r / torch.linalg.norm(r, dim = -1)[:,None]

    # Hessenberg matrix
    H = torch.zeros((nbatch, maxiter + 1, maxiter))

    # Residual norm
    nR = torch.linalg.norm(r, dim = -1)

    # Unit vector
    e1 = torch.zeros((nbatch,maxiter+1))
    e1[:,0] = nR

    # Start iterating
    for k in range(maxiter):
        # A \cdot v
        w = A.bmm(V[...,:,k].unsqueeze(-1)).squeeze(-1)
        
        # Orthogonalize
        for j in range(k+1):
            H[...,j,k] = V[...,:,j].unsqueeze(-2).bmm(w.unsqueeze(-1)).squeeze(-2).squeeze(-1)
            w -= H[...,j,k].unsqueeze(-1) * V[...,:,j]

        # Next Arnoldi vector
        H[...,k+1,k] = torch.linalg.norm(w, axis = -1)
        V[...,:,k+1] = w / H[...,k+1, k][:,None]
        
        # If it's time, check the new residual value
        if (k % check == 0 and k != 0) or k == maxiter:
            # Project onto our basis
            y, _, _, _ = torch.linalg.lstsq(H[...,:k+1,:k], e1[...,:k+1], rcond = None)
            # Calculate the value of x
            x = V[...,:,:k].bmm(y.unsqueeze(-1)).squeeze(-1) + x0
            # Calculate the residual and the norm of the residual
            r = b - A.bmm(x.unsqueeze(-1)).squeeze(-1)
            nRc = torch.linalg.norm(r, dim = -1)
            # Check for convergence
            if torch.all(nRc < tol):
                break

    return x
