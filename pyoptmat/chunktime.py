import warnings

import torch
import numpy as np

def newton_raphson_chunk(fn, x0, solver, rtol=1e-6, atol=1e-10, miter=100):
    """
    Solve a nonlinear system with Newton's method with a ChunkTimeOperatorSolverContext
    context manager.  Return the
    solution and the last Jacobian

    Args:
      fn (function):        function that returns R, J, and the solver context
      x0 (torch.tensor):    starting point
      solver (ChunkTimeOperatorSolverContext): solver context

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

class ChunkTimeOperatorSolverContext:
    """
    Context manager for solving our special ChunkTimeOperator system
    using GMRES with preconditioner reuse.

    Args:
        solve_method:   one of "dense", "direct", or "gmres"
    """
    def __init__(self, solve_method):
        if solve_method not in ["dense", "direct", "gmres"]:
            raise ValueError("Solve method must be one of dense, direct, or gmres")
        self.solve_method = solve_method
        self.J = None

    def solve(self, J, R):
        """
        Actually solve Jx = R

        Args:
            J (ChunkTimeOperator):  matrix operator
            R (torch.tensor):       right hand side
        """
        if self.solve_method == "dense":
            return self.solve_dense(J, R)
        elif self.solve_method == "direct":
            return self.solve_direct(J, R)
        else:
            raise RuntimeError("Unknown solver method...")

    def solve_dense(self, J, R):
        """
        Highly inefficient solve where we first convert to a dense tensor

        Args:
            J (ChunkTimeOperator):  matrix operator
            R (torch.tensor):       right hand side
        """
        return torch.linalg.solve_ex(J.to_diag().to_dense(), R)[0]

    def solve_direct(self, J, R):
        """
        Solve with Thomas's algorithm applied to our special case

        Args:
            J (ChunkTimeOperator):  matrix operator
            R (torch.tensor):       right hand side
        """
        lu, pivots, _ = torch.linalg.lu_factor_ex(J.diag)
        return block_thomas(lu, pivots, J, R)

def block_thomas(lu, pivots, J, R):
    """
    Specialized version of Thomas's algorithm for our block banded 
    structure.

    Args:
        lu:                     lu from torch.linalg.factor_ex
        pivots:                 pivots from torch.linalg.factor_ex
        J (ChunkTimeOperator):  actual operator
        R (torch.tensor):       right hand side
    """
    y = torch.empty_like(R)
    i = 0
    s = J.sblk
    y[:,i*s:(i+1)*s] = torch.linalg.lu_solve(lu[i], pivots[i], R[:,i*s:(i+1)*s].unsqueeze(-1)).squeeze(-1)
    for i in range(1, J.nblk):
        y[:,i*s:(i+1)*s] = torch.linalg.lu_solve(lu[i], pivots[i], (R[:,i*s:(i+1)*s] + y[:,(i-1)*s:i*s]).unsqueeze(-1)).squeeze(-1)

    return y
        

class ChunkTimeOperator:
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
    def __init__(self, diag):
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
                    torch.eye(self.sblk, device = self.device).expand(self.nblk-1,self.sbat,-1,-1)
                ], 
                [0, -1])

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
