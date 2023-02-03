import warnings

import torch
import numpy as np
from math import log2

def gmres_operators(Av, b, x0 = None, Mv = lambda x: x, tol = 1e-10, 
        maxiter = 100, check = 5, eps = 1.0e-15, return_info = False):
    """
    Solve a linear system of equations using GMRES.

    Args:
        Av (function):  function providing $A \cdot v$
        b (torch.tensor): right hand side

    Keyword Args:
        x0 (torch.tensor): initial guess, defaults to zero
        Mv (function): function provide $M^{-1} \cdot v$
        tol (float): absolute solver tolerance
        maxiter (int): maximum number of iterations
        check (int): check residual every check iterations
        eps (float): small number to avoid zero vectors
        return_info (bool): if True return additional information
            on the solve process

    Returns:
        x (torch.tensor): result
        info (dict): dictionary with
            'niter' number of iterations required
            'max_res' maximum norm of the residual
    """
    # Should fix code at some point to treat batch dimensions right
    nbatch = b.shape[0]

    # Initial guess
    if x0 is None:
        x0 = torch.zeros_like(b)
    x = x0

    # Initial residual
    r = b - Av(x)

    # Basis
    V = torch.zeros((nbatch, b.shape[-1], maxiter + 1), device=b.device)
    V[..., :, 0] = r / torch.linalg.norm(r, dim=-1)[:, None]

    # Preconditioned basis
    Z = torch.zeros((nbatch, b.shape[-1], maxiter + 1), device=b.device)
    Z[..., :, 0] = Mv(V[..., :, 0])

    # Hessenberg matrix
    H = torch.zeros((nbatch, maxiter + 1, maxiter), device=b.device)

    # Residual norm
    nR = torch.linalg.vector_norm(r, dim=-1)

    # Check for an initial zero
    if torch.all(nR < tol):
        if return_info:
            return x0, {'niter': 0, 'max_res': 0.0}
        return x0

    # Unit vector
    e1 = torch.zeros((nbatch, maxiter + 1), device=b.device)
    e1[:, 0] = nR

    # Start iterating
    for k in range(maxiter):
        # A \cdot v
        w = Av(Z[..., :, k].unsqueeze(-1)).squeeze(-1)

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
        Z[..., :, k + 1] = Mv(V[..., :, k + 1])

        # If it's time, check the new residual value
        if (k + 1) % check == 0 or (k + 1) == maxiter:
            # Project onto our basis
            y, _, _, _ = torch.linalg.lstsq(
                H[..., : k + 1, :k], e1[..., : k + 1], rcond=None
            )
            # Calculate the value of x
            x = Z[..., :, :k].bmm(y.unsqueeze(-1)).squeeze(-1) + x0
            # Calculate the residual and the norm of the residual
            r = b - Av(x.unsqueeze(-1)).squeeze(-1)
            nRc = torch.linalg.vector_norm(r, dim=-1)
            # Check for convergence
            if torch.all(nRc < tol):
                break
    
    if return_info:
        return x, {"niter": k, "max_res": torch.max(nRc)}
    return x

def newton_raphson_chunk(fn, x0, solver, rtol=1e-6, atol=1e-10, miter=100):
    """
    Solve a nonlinear system with Newton's method with a BackwardEulerChunkTimeOperatorSolverContext
    context manager.  Return the
    solution and the last Jacobian

    Args:
      fn (function):        function that returns R, J, and the solver context
      x0 (torch.tensor):    starting point
      solver (BackwardEulerChunkTimeOperatorSolverContext): solver context

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
    x = x0
    R, J = fn(x)

    nR = torch.norm(R, dim=-1)
    nR0 = nR
    i = 0

    while (i < miter) and torch.any(nR > atol) and torch.any(nR / nR0 > rtol):
        x -= solver.solve(J, R)
        R, J = fn(x)
        nR = torch.norm(R, dim=-1)
        i += 1
    
    if i == miter:
        warnings.warn("Implicit solve did not succeed.  Results may be inaccurate...")
    
    return x, J

class Factorization:
    """
    A generic factorization object used to efficiently calculate $A^{-1} \cdot y$

    Args:
        A (BackwardEulerChunkTimeOperator): base operator
    """
    def __init__(self, A):
        self.nblk = A.nblk
        self.sbat = A.sbat
        self.sblk = A.sblk

        self.shape = A.shape

        self._setup_factorization(A.diag)

class ThomasFactorization(Factorization):
    """
    Manages the data needed to solve our special system via Thomas factorization

    Args:
        A (BackwardEulerChunkTimeOperator): base operator
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def dot(self, v):
        """
        Complete the backsolve for a given right hand side

        Args:
            v (torch.tensor): tensor of shape (sbat, sblk*nblk)
        """
        y = torch.empty_like(v)
        i = 0
        s = self.sblk
        y[:,i*s:(i+1)*s] = torch.linalg.lu_solve(self.lu[i], self.pivots[i], v[:,i*s:(i+1)*s].unsqueeze(-1)).squeeze(-1)
        for i in range(1, self.nblk):
            y[:,i*s:(i+1)*s] = torch.linalg.lu_solve(self.lu[i], self.pivots[i], (v[:,i*s:(i+1)*s] + y[:,(i-1)*s:i*s]).unsqueeze(-1)).squeeze(-1)

        return y

    def _setup_factorization(self, diag):
        """
        Form the factorization...

        Args:
            diag (torch.tensor): diagonal blocks of shape (nblk, sbat, sblk, sblk)
        """
        self.lu, self.pivots, _ = torch.linalg.lu_factor_ex(diag)

class DirectFactorization(Factorization):
    """
    Manages the data needed to solve our special system via direct factorization

    Args:
        diag (torch.tensor): diagonal blocks of shape (nblk,sbat,sblk,sblk)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dot(self, v):
        """
        Complete the backsolve for a given right hand side

        Args:
            v (torch.tensor): tensor of shape (sbat, sblk*nblk)
        """
        vp = self.operator.bmm(v.unsqueeze(-1)).squeeze(-1).view(self.sbat,self.nblk, self.sblk).transpose(0,1)
        return torch.linalg.lu_solve(self.lu, self.pivots, vp.unsqueeze(-1)).squeeze(-1).transpose(0,1).flatten(start_dim = 1)

    def _setup_factorization(self, diag):
        """
        Form the factorization...

        Args:
            diag (torch.tensor): diagonal blocks of shape (nblk, sbat, sblk, sblk)
        """
        factorization = torch.zeros_like(diag)
        self.operator = torch.zeros((self.sbat, self.nblk, self.sblk, self.nblk, self.sblk), device = diag.device, dtype = diag.dtype)
        factorization[0] = torch.eye(self.sblk).unsqueeze(0).expand(self.sbat, self.sblk, self.sblk)
        
        for i in range(self.nblk):
            self.operator[:,i:,:,i] = factorization[i].unsqueeze(1).expand(self.sbat, self.nblk - i, self.sblk, self.sblk)
            factorization[i] = torch.bmm(factorization[i], diag[i])
            if i != self.nblk - 1:
                factorization[i+1] = factorization[i]

        self.operator = self.operator.view(self.sbat, self.nblk*self.sblk, self.nblk*self.sblk)
        self.lu, self.pivots, _ = torch.linalg.lu_factor_ex(factorization) 


class BackwardEulerChunkTimeOperator:
    """
    A batched block banded matrix of the form:
    A1   0   0   0   ...   0
    -I   A2  0   0   ...   0
    0    -I  A3  0   ...   0
    .    .   .   .   ...   0
    .    .   .   .   ...   0
    0    0   0   0    -I   An
    
    that is, a blocked banded system with the main
    diagonal some arbitrary list of tensor blocks and the
    1st block lower diagonal minus the identity.

    We use the following sizes:
        nblk:   number of blocks in the square matrix
        sblk:   size of each block
        sbat:   batch size

    Args:
        data (torch.tensor): tensor of shape (nblk,sbat,sblk,sblk)
            storing the nblk diagonal blocks
    """
    def __init__(self, diag, factorization = ThomasFactorization):
        self.diag = diag

        self.nblk = self.diag.shape[0]
        self.sbat = self.diag.shape[1]
        self.sblk = self.diag.shape[3]

    @property
    def dtype(self):
        """
        dtype, which is just the dtype of self.diag
        """
        return self.diag.dtype

    @property
    def device(self):
        """
        device, which is just the device of self.diag
        """
        return self.diag.device

    @property
    def n(self):
        """
        Size of the unbatched square matrix
        """
        return self.nblk * self.sblk

    @property
    def shape(self):
        """
        Logical shape of the dense array
        """
        return (self.sbat, self.n, self.n)

    def to_diag(self):
        """
        Convert to a SquareBatchedBlockDiagonalMatrix, for testing
        or legacy purposes
        """
        return SquareBatchedBlockDiagonalMatrix(
                [
                    self.diag,
                    -torch.eye(self.sblk, device = self.device).expand(self.nblk-1,self.sbat,-1,-1)
                ], 
                [0, -1])

    def dot(self, v):
        """
        $A \cdot v$ in an efficient manner

        Args:
            v (torch.tensor):   batch of vectors
        """
        # Reshaped v 
        vp = v.view(self.sbat, self.nblk, self.sblk).transpose(0,1)
        
        b = torch.bmm(self.diag.view(-1,self.sblk,self.sblk),
                vp.reshape(self.sbat*self.nblk,self.sblk).unsqueeze(-1)).squeeze(-1).view(self.nblk,self.sbat,self.sblk)
        b[1:] -= vp[:-1]

        return b.transpose(1,0).flatten(start_dim=1)

class ChunkTimeOperatorSolverContext:
    """
    Context manager for solving our special BackwardEulerChunkTimeOperator system
    using GMRES with preconditioner reuse.

    Args:
        solve_method:   one of "dense", "direct", or "gmres"

    Keyword Args:
        gmres_reuse_iters (int): number of gmres iterations to trigger a new preconditioner
        gmres_tol (float): absolute tolerance for GMRES
        gmres_miter (int): number of iterations to allow for GMRES
        gmres_check (int): interval for checking residual in GMRES
    """
    def __init__(self, solve_method, gmres_reuse_iters = 50,
            gmres_tol = 1.0e-10, gmres_miter = 100,
            gmres_check = 5, factorization = ThomasFactorization):

        if solve_method not in ["dense", "direct", "gmres"]:
            raise ValueError("Solve method must be one of dense, direct, or gmres")
        self.solve_method = solve_method

        self.gmres_reuse_iters = gmres_reuse_iters
        self.gmres_tol = gmres_tol
        self.gmres_miter = gmres_miter
        self.gmres_check = gmres_check
        self.factorization = factorization

        self.M = None
        self.last_iters = 0

    def solve(self, J, R):
        """
        Actually solve Jx = R

        Args:
            J (BackwardEulerChunkTimeOperator):  matrix operator
            R (torch.tensor):       right hand side
        """
        if self.solve_method == "dense":
            return self.solve_dense(J, R)
        elif self.solve_method == "direct":
            return self.solve_direct(J, R)
        elif self.solve_method == "gmres":
            return self.solve_gmres(J, R)
        else:
            raise RuntimeError("Unknown solver method...")

    def solve_dense(self, J, R):
        """
        Highly inefficient solve where we first convert to a dense tensor

        Args:
            J (BackwardEulerChunkTimeOperator):  matrix operator
            R (torch.tensor):       right hand side
        """
        return torch.linalg.solve_ex(J.to_diag().to_dense(), R)[0]

    def solve_direct(self, J, R):
        """
        Solve with a direct factorization

        Args:
            J (BackwardEulerChunkTimeOperator):  matrix operator
            R (torch.tensor):       right hand side
        """
        M = self.factorization(J)
        return M.dot(R)

    def solve_gmres(self, J, R):
        """
        Solve with GMRES using a preconditioner reuse heuristic

        Args:
            J (BackwardEulerChunkTimeOperator):  matrix operator
            R (torch.tensor):       right hand side
        """
        if (
                self.M is None or 
                self.M.shape != J.shape or
                self.last_iters > self.gmres_reuse_iters
            ):
            self.M = self.factorization(J)
        
        b, info = gmres_operators(lambda x: J.dot(x), R, 
                Mv = lambda x: self.M.dot(x),
                tol = self.gmres_tol,
                maxiter = self.gmres_miter,
                check = self.gmres_check,
                return_info = True)

        self.last_iters = info['niter']

        return b

class SquareBatchedBlockDiagonalMatrix:
    """
    A batched block diagonal matrix of the type

    A1  B1  0   0
    C1  A2  B2  0
    0   C2  A3  B3
    0   0   C3  A4

    where the matrix has diagonal blocks of non-zeros

    Additionally, this matrix is batched.

    We use the following sizes:
        nblk:     number of blocks in the each direction
        sblk:     size of each block
        sbat:     batch size

    Args:
        data (list of tensors):     list of tensors of length ndiag.
                                    Each tensor
                                    has shape (nblk-|d|,sbat,sblk,sblk)
                                    where d is the diagonal number 
                                    provided in the next input
        diags (list of ints):       list of ints of length ndiag.
                                    Each entry gives the diagonal 
                                    for the data in the corresponding
                                    tensor.  These values d can 
                                    range from -(n-1) to (n-1)
    """
    def __init__(self, data, diags):
        # We will want this in order later
        iargs = np.argsort(diags)

        self.data = [data[i] for i in iargs]
        self.diags = [diags[i] for i in iargs]

        self.nblk = self.data[0].shape[0]+abs(self.diags[0])
        self.sbat = self.data[0].shape[1]
        self.sblk = self.data[0].shape[-1]
    
    @property
    def dtype(self):
        """
        dtype, as reported by the first entry in self.data
        """
        return self.data[0].dtype

    @property
    def device(self):
        """
        device, as reported by the first entry in self.device
        """
        return self.data[0].device

    @property
    def n(self):
        """
        Size of the unbatched square matrix
        """
        return self.nblk * self.sblk

    @property
    def shape(self):
        """
        Logical shape of the dense array
        """
        return (self.sbat, self.n, self.n)

    @property
    def nnz(self):
        """
        Number of logical non-zeros (not counting the batch dimension)
        """
        return sum(self.data[i].shape[0] * self.sblk * self.sblk for i in range(len(self.diags)))

    def to_dense(self):
        """
        Convert the representation to a dense tensor
        """
        A = torch.zeros(*self.shape, dtype = self.dtype, 
                        device = self.device)
        
        # There may be a more clever way than for loops, but for now
        for d, data in zip(self.diags, self.data):
            for k in range(self.nblk - abs(d)):
                if d <= 0:
                    i = k - d
                    j = k
                else:
                    i = k
                    j = k + d
                A[:,i*self.sblk:(i+1)*self.sblk,j*self.sblk:(j+1)*self.sblk] = data[k]

        return A

    def to_batched_coo(self):
        """
        Convert to a torch sparse batched COO tensor

        This is done in a weird way.  torch recognizes "batch" dimensions at
        the start of the tensor and "dense" dimensions at the end (with "sparse"
        dimensions in between).  batch dimensions can/do have difference indices,
        dense dimensions all share the same indices.  We have the latter situation
        so this is setup as a tensor with no "batch" dimensions, 2 "sparse" dimensions,
        and 1 "dense" dimension.  So it will be the transpose of the shape of the 
        to_dense function.
        """
        inds = torch.zeros(2,self.nnz)
        data = torch.zeros(self.nnz, self.sbat, dtype = self.dtype, 
                           device = self.device)
        
        # Order doesn't matter, nice!
        c = 0
        chunk = self.sblk * self.sblk
        for d, bdata in zip(self.diags, self.data):
            for i in range(bdata.shape[0]):
                data[c:c+chunk] = bdata[i].flatten(start_dim=1).t()
                
                offset = (i + abs(d)) * self.sblk
                
                if d < 0:
                    roffset = offset
                    coffset = i * self.sblk
                else:
                    roffset = i * self.sblk
                    coffset = offset

                inds[0,c:c+chunk] = torch.repeat_interleave(
                        torch.arange(0,self.sblk,dtype=torch.int64,device=self.device
                                     ).unsqueeze(-1), self.sblk, -1).flatten() + roffset
                inds[1,c:c+chunk] = torch.repeat_interleave(
                        torch.arange(0,self.sblk,dtype=torch.int64,device=self.device
                                     ).unsqueeze(0), self.sblk, 0).flatten() + coffset

                c += chunk

        return torch.sparse_coo_tensor(inds, data, 
                                       dtype = self.dtype, 
                                       device = self.device,
                                       size = (self.n, self.n, self.sbat)).coalesce()

    def to_unrolled_csr(self):
        """
        Return a list of CSR tensors with length equal to the batch size

        """
        coo = self.to_batched_coo()
        return [torch.sparse_coo_tensor(coo.indices(), coo.values()[:,i]).to_sparse_csr() for i in range(self.sbat)]
