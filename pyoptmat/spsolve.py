import torch
import numpy as np

def dense_solve(A, b):
    """
    Solve a sparse linear system by first converting the matrix to dense

    Args:
        A (SquareBatchedBlockDiagonalMatrix): batch of matrices
        b (torch.tensor): batch of tensors
    """
    return torch.linalg.solve_ex(A.to_dense(), b)[0]

def newton_raphson_sparse(fn, x0, solver=dense_solve, rtol=1e-6, atol=1e-10, miter=100):
    """
    Solve a nonlinear system with Newton's method with sparse operators.  Return the
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


