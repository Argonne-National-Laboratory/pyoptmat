"""
  A few common functions related to solving nonlinear and linear systems
  of equations.  These functions exist to "torchify" these common tasks.
  As always, the routines accept batched input.
"""

import warnings
import torch


def newton_raphson_bt(
    fn, x0, linsolver="lu", rtol=1e-6, atol=1e-10, miter=100, max_bt=5
):
    """
    Solve a nonlinear system with Newton's method.  Return the
    solution and the last Jacobian

    Args:
      fn (function):        function that returns the residual and Jacobian
      x0 (torch.tensor):    starting point

    Keyword Args:
      linsolver (string or function):   method to use to solve the linear system,
                                        strain options are "diag" or "lu".
                                        Defaults to "lu".
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


def lu_linear_solve(A, b, return_iters=False):
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


class NoOp:
    """
    Linear operator that does nothing
    """

    def __init__(self):
        pass

    def dot(self, x):
        """
        Batched matrix-vector product

        Here just returns the input unmodified

        Args:
            x (torch.tensor):       input to linear operator

        Returns:
            y (torch.tensor):       result of the linear operator
        """
        return x


class JacobiPreconitionerOperator:
    """
    Jacobi preconditioning based on another matrix

    Args:
        A (torch.tensor):       operator to take diagonal from
    """

    def __init__(self, A):
        self.diag_A = A.diagonal(dim1=-2, dim2=-1)

    def dot(self, x):
        """
        Batched matrix-vector product

        This object returns x_i / A_ii

        Args:
            x (torch.tensor):       input to linear operator

        Returns:
            y (torch.tensor):       result of the linear operator
        """
        return x / self.diag_A


class LUPreconitionerOperator:
    """
    Full LU preconditioning based on another matrix

    Args:
        A (torch.tensor):       operator to LU factorize
    """

    def __init__(self, A):
        self.LU, self.pivots = torch.linalg.lu_factor(A)

    def dot(self, x):
        """
        Batched matrix-vector product

        This object does the backsolves to calculate (LU)^-1 * x

        Args:
            x (torch.tensor):       input to linear operator

        Returns:
            y (torch.tensor):       result of the linear operator
        """
        return torch.linalg.lu_solve(self.LU, self.pivots, x.unsqueeze(-1)).squeeze(-1)


def lu_gmres(A, *args, **kwargs):
    """
    Solve a linear system of equations using LU preconditioned GMRES

    This is an entirely pointless operation, but useful in testing

    Args:
        A (torch.tensor):   black matrix
        b (torch.tensor):   black RHS

    Keyword Args:
        x0 (torch.tensor):  initial guess, defaults to zeros
        M (LinearOperator): preconditioning operator
        tol (float):        absolute tolerance for solve
        maxiter (int):      maximum number of iterations
        check (int):        how often to check the residual

    Returns:
        x (torch.tensor):   block results
    """
    M = LUPreconitionerOperator(A)
    kwargs.pop("check", None)
    return gmres(A, *args, M=M, check=1, **kwargs)


def jacobi_gmres(A, *args, **kwargs):
    """
    Solve a linear system of equations using Jacobi preconditioned GMRES

    Args:
        A (torch.tensor):   black matrix
        b (torch.tensor):   black RHS

    Keyword Args:
        x0 (torch.tensor):  initial guess, defaults to zeros
        M (LinearOperator): preconditioning operator
        tol (float):        absolute tolerance for solve
        maxiter (int):      maximum number of iterations
        check (int):        how often to check the residual

    Returns:
        x (torch.tensor):   block results
    """
    M = JacobiPreconitionerOperator(A)
    return gmres(A, *args, M=M, **kwargs)


def gmres(
    A,
    b,
    x0=None,
    M=NoOp(),
    tol=1e-10,
    maxiter: int = 100,
    check=5,
    return_iters=False,
    eps=1.0e-15,
):
    """
    Solve a linear system of equations using GMRES

    Args:
        A (torch.tensor):   black matrix
        b (torch.tensor):   black RHS

    Keyword Args:
        x0 (torch.tensor):      initial guess, defaults to zeros
        M (LinearOperator):     preconditioning operator
        tol (float):            absolute tolerance for solve
        maxiter (int):          maximum number of iterations
        check (int):            how often to check the residual
        return_iters (bool):    return iteration count along with solution
        eps (float):            small number to avoid zero vectors

    Returns:
        x (torch.tensor):   block results
        k (int, optional):  iteration count
    """
    # Should fix code at some point to treat batch dimensions right
    nbatch = A.shape[0]

    # Initial guess
    if x0 is None:
        x0 = torch.zeros_like(b)
    x = x0

    # Initial residual
    r = b - A.bmm(x.unsqueeze(-1)).squeeze(-1)

    # Basis
    V = torch.zeros((nbatch, b.shape[-1], maxiter + 1), device=A.device)
    V[..., :, 0] = r / torch.linalg.norm(r, dim=-1)[:, None]

    # Preconditioned basis
    Z = torch.zeros((nbatch, b.shape[-1], maxiter + 1), device=A.device)
    Z[..., :, 0] = M.dot(V[..., :, 0])

    # Hessenberg matrix
    H = torch.zeros((nbatch, maxiter + 1, maxiter), device=A.device)

    # Residual norm
    nR = torch.linalg.vector_norm(r, dim=-1)

    # Check for an initial zero
    if torch.all(nR < tol):
        if return_iters:
            return x0, 0
        return x0

    # Unit vector
    e1 = torch.zeros((nbatch, maxiter + 1), device=A.device)
    e1[:, 0] = nR

    # Start iterating
    for k in range(maxiter):
        # A \cdot v
        w = A.bmm(Z[..., :, k].unsqueeze(-1)).squeeze(-1)

        # Orthogonalize
        for j in range(k + 1):
            H[..., j, k] = (
                V[..., :, j].unsqueeze(-2).bmm(w.unsqueeze(-1)).squeeze(-2).squeeze(-1)
            )
            w -= H[..., j, k].unsqueeze(-1) * V[..., :, j]

        # Next Arnoldi vector
        H[..., k + 1, k] = torch.linalg.vector_norm(w, dim=-1) + eps
        V[..., :, k + 1] = w / H[..., k + 1, k][:, None]

        # Preconditioned
        Z[..., :, k + 1] = M.dot(V[..., :, k + 1])

        # If it's time, check the new residual value
        if (k + 1) % check == 0 or (k + 1) == maxiter:
            # Project onto our basis
            y, _, _, _ = torch.linalg.lstsq(
                H[..., : k + 1, :k], e1[..., : k + 1], rcond=None
            )
            # Calculate the value of x
            x = Z[..., :, :k].bmm(y.unsqueeze(-1)).squeeze(-1) + x0
            # Calculate the residual and the norm of the residual
            r = b - A.bmm(x.unsqueeze(-1)).squeeze(-1)
            nRc = torch.linalg.vector_norm(r, dim=-1)
            # Check for convergence
            if torch.all(nRc < tol):
                break

    if return_iters:
        return x, k
    return x


class PreconditionerReuseNonlinearSolver:
    """
    Maintains preconditioner reuse through subsequent linear solves
    """

    def __init__(self, nonlinear_solver=newton_raphson):
        self.nonlinear_solver = nonlinear_solver

        self.stored_preconditioner = None

    def _update_preconditioner(self, J):
        self.stored_preconditioner = LUPreconitionerOperator(J)

    def solve(self, fn, x0, rtol=1e-6, atol=1e-10, miter=100, ltol=1e-10):
        """
        Solve the nonlinear system
        """
        x = x0
        R, J = fn(x)

        if not self.stored_preconditioner:
            self._update_preconditioner(J)

        nR = torch.norm(R, dim=-1)
        nR0 = nR
        i = 0

        l_miter = J.shape[-1] + 1
        l_check = max(1, (J.shape[-1] + 1) // 10)
        l_refactor = (J.shape[-1] * 2) // 3

        while (i < miter) and torch.any(nR > atol) and torch.any(nR / nR0 > rtol):
            dx, niter = gmres(
                J,
                R,
                x0=x,
                M=self.stored_preconditioner,
                maxiter=l_miter,
                check=l_check,
                return_iters=True,
                tol=ltol,
            )
            x -= dx
            R, J = fn(x)
            if niter > l_refactor:
                self._update_preconditioner(J)
            nR = torch.norm(R, dim=-1)
            i += 1

        if i == miter:
            warnings.warn(
                "Implicit solve did not succeed.  Results may be inaccurate..."
            )

        return x, J
