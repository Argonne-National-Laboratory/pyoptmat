# pylint: disable=abstract-method

"""
    Functions and objects to help with blocked/chunked time integration.

    These include:
        1. Sparse matrix classes for banded systems
        2. General sparse matrix classes
        3. Specialized solver routines working with banded systems
"""

import warnings

import torch
import torch.jit
import numpy as np


def newton_raphson_chunk(fn, x0, solver, rtol=1e-6, atol=1e-10, miter=100):
    """
    Solve a nonlinear system with Newton's method with a tensor for a
    BackwardEuler type chunking operator context manager.

    Args:
      fn (function):        function that returns R, J, and the solver context
      x0 (torch.tensor):    starting point
      solver (ChunkTimeOperatorSolverContext): solver context

    Keyword Args:
      rtol (float):         nonlinear relative tolerance
      atol (float):         nonlinear absolute tolerance
      miter (int):          maximum number of nonlinear iterations

    Returns:
      torch.tensor:         solution to system of equations
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

    return x


class BidiagonalOperator(torch.nn.Module):
    """
    An object working with a Batched block diagonal operator of the type

    .. math::

        \\begin{bmatrix}
        A_1 & 0 & 0 & 0 & \\cdots  & 0\\\\
        B_1 & A_2 & 0 & 0 & \\cdots & 0\\\\
        0 & B_2 & A_3 & 0 & \\cdots & 0\\\\
        \\vdots & \\vdots & \\ddots & \\ddots & \\ddots  & \\vdots \\\\
        0 & 0 & 0 & B_{n-2} & A_{n-1} & 0\\\\
        0 & 0 & 0 & 0 & B_{n-1} & A_n
        \\end{bmatrix}

    that is, a blocked banded system with the main
    diagonal and the first lower diagonal filled

    We use the following sizes:
        - nblk:   number of blocks in the square matrix
        - sblk:   size of each block
        - sbat:   batch size

    Args:
        A (torch.tensor): tensor of shape (nblk,sbat,sblk,sblk)
            storing the nblk main diagonal blocks
        B (torch.tensor): tensor of shape (nblk-1,sbat,sblk,sblk)
            storing the nblk-1 off diagonal blocks
    """

    def __init__(self, A, B, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = A
        self.B = B
        self.nblk = A.shape[0]
        self.sbat = A.shape[1]
        self.sblk = A.shape[2]

    @property
    def dtype(self):
        """
        dtype, which is just the dtype of self.diag
        """
        return self.A.dtype

    @property
    def device(self):
        """
        device, which is just the device of self.diag
        """
        return self.A.device

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


class BidiagonalThomasFactorization(BidiagonalOperator):
    """
    Manages the data needed to solve our bidiagonal system via Thomas
    factorization

    Args:
        A (torch.tensor): tensor of shape (nblk,sbat,sblk,sblk) with the main diagonal
        B (torch.tensor): tensor of shape (nblk-1,sbat,sblk,sblk) with the off diagonal
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._setup_factorization()

    def forward(self, v):
        """
        Complete the backsolve for a given right hand side

        Args:
            v (torch.tensor): tensor of shape (sbat, sblk*nblk)
        """
        y = torch.empty_like(v)
        i = 0
        s = self.sblk
        y[:, i * s : (i + 1) * s] = torch.linalg.lu_solve(
            self.lu[i], self.pivots[i], v[:, i * s : (i + 1) * s].unsqueeze(-1)
        ).squeeze(-1)
        # The .clone() here really makes no sense to me, but torch assures me it is necessary
        for i in range(1, self.nblk):
            y[:, i * s : (i + 1) * s] = torch.linalg.lu_solve(
                self.lu[i],
                self.pivots[i],
                v[:, i * s : (i + 1) * s].unsqueeze(-1)
                - torch.bmm(
                    self.B[i - 1], y[:, (i - 1) * s : i * s].clone().unsqueeze(-1)
                ),
            ).squeeze(-1)

        return y

    def _setup_factorization(self):
        """
        Form the factorization...

        Args:
            diag (torch.tensor): diagonal blocks of shape (nblk, sbat, sblk, sblk)
        """
        self.lu, self.pivots, _ = torch.linalg.lu_factor_ex(self.A)


class BidiagonalForwardOperator(BidiagonalOperator):
    """
    A batched block banded matrix of the form:

    .. math::

        \\begin{bmatrix}
        A_1 & 0 & 0 & 0 & \\cdots  & 0\\\\
        B_1 & A_2 & 0 & 0 & \\cdots & 0\\\\
        0 & B_2 & A_3 & 0 & \\cdots & 0\\\\
        \\vdots & \\vdots & \\ddots & \\ddots & \\ddots  & \\vdots \\\\
        0 & 0 & 0 & B_{n-2} & A_{n-1} & 0\\\\
        0 & 0 & 0 & 0 & B_{n-1} & A_n
        \\end{bmatrix}

    that is, a blocked banded system with the main
    diagonal and first lower block diagonal filled

    We use the following sizes:
        - nblk: number of blocks in the square matrix
        - sblk: size of each block
        - sbat: batch size

    Args:
        A (torch.tensor): tensor of shape (nblk,sbat,sblk,sblk)
            storing the nblk diagonal blocks
        B (torch.tensor): tensor of shape (nblk-1,sbat,sblk,sblk)
            storing the nblk-1 off diagonal blocks
    """

    def to_diag(self):
        """
        Convert to a SquareBatchedBlockDiagonalMatrix, for testing
        or legacy purposes
        """
        return SquareBatchedBlockDiagonalMatrix([self.A, self.B], [0, -1])

    def forward(self, v):
        """
        :math:`A \\cdot v` in an efficient manner

        Args:
            v (torch.tensor):   batch of vectors
        """
        # Reshaped v
        vp = (
            v.view(self.sbat, self.nblk, self.sblk)
            .transpose(0, 1)
            .reshape(self.sbat * self.nblk, self.sblk)
            .unsqueeze(-1)
        )

        # Actual calculation
        b = torch.bmm(self.A.view(-1, self.sblk, self.sblk), vp)
        b[self.sbat :] += torch.bmm(
            self.B.view(-1, self.sblk, self.sblk), vp[: -self.sbat]
        )

        return (
            b.squeeze(-1)
            .view(self.nblk, self.sbat, self.sblk)
            .transpose(1, 0)
            .flatten(start_dim=1)
        )

    def inverse(self):
        """
        Return an inverse operator
        """
        return BidiagonalThomasFactorization(self.A, self.B)


class ChunkTimeOperatorSolverContext:
    """
    Context manager for solving sparse chunked time systems

    Args:
        solve_method:   one of "dense" or "direct"
    """

    def __init__(self, solve_method):

        if solve_method not in ["dense", "direct"]:
            raise ValueError("Solve method must be one of dense or direct")
        self.solve_method = solve_method

    def solve(self, J, R):
        """
        Actually solve Jx = R

        Args:
            J (BidiagonalForwardOperator):  matrix operator
            R (torch.tensor):       right hand side
        """
        if self.solve_method == "dense":
            return self.solve_dense(J, R)
        if self.solve_method == "direct":
            return self.solve_direct(J, R)
        raise RuntimeError("Unknown solver method...")

    def solve_dense(self, J, R):
        """
        Highly inefficient solve where we first convert to a dense tensor

        Args:
            J (BidiagonalForwardOperator):  matrix operator
            R (torch.tensor):       right hand side
        """
        return torch.linalg.solve_ex(J.to_diag().to_dense(), R)[0]

    def solve_direct(self, J, R):
        """
        Solve with a direct factorization

        Args:
            J (BidiagonalForwardOperator):  matrix operator
            R (torch.tensor):       right hand side
        """
        M = J.inverse()
        return M(R)


class SquareBatchedBlockDiagonalMatrix:
    """
    A batched block diagonal matrix of the type

    .. math::

        \\begin{bmatrix}
        A_1 & B_1 & 0 & 0\\\\
        C_1 & A_2 & B_2 & 0 \\\\
        0 & C_2 & A_3 & B_3\\\\
        0 & 0 & C_3 & A_4
        \\end{bmatrix}

    where the matrix has diagonal blocks of non-zeros and
    can have arbitrary numbers of filled diagonals

    Additionally, this matrix is batched.

    We use the following sizes:
        - nblk: number of blocks in the each direction
        - sblk: size of each block
        - sbat: batch size

    Args:
        data (list of tensors):     list of tensors of length ndiag.
                                    Each tensor
                                    has shape :code:`(nblk-abs(d),sbat,sblk,sblk)`
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

        self.nblk = self.data[0].shape[0] + abs(self.diags[0])
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
        return sum(
            self.data[i].shape[0] * self.sblk * self.sblk
            for i in range(len(self.diags))
        )

    def to_dense(self):
        """
        Convert the representation to a dense tensor
        """
        A = torch.zeros(*self.shape, dtype=self.dtype, device=self.device)

        # There may be a more clever way than for loops, but for now
        for d, data in zip(self.diags, self.data):
            for k in range(self.nblk - abs(d)):
                if d <= 0:
                    i = k - d
                    j = k
                else:
                    i = k
                    j = k + d
                A[
                    :,
                    i * self.sblk : (i + 1) * self.sblk,
                    j * self.sblk : (j + 1) * self.sblk,
                ] = data[k]

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
        inds = torch.zeros(2, self.nnz)
        data = torch.zeros(self.nnz, self.sbat, dtype=self.dtype, device=self.device)

        # Order doesn't matter, nice!
        c = 0
        chunk = self.sblk * self.sblk
        for d, bdata in zip(self.diags, self.data):
            for i in range(bdata.shape[0]):
                data[c : c + chunk] = bdata[i].flatten(start_dim=1).t()

                offset = (i + abs(d)) * self.sblk

                if d < 0:
                    roffset = offset
                    coffset = i * self.sblk
                else:
                    roffset = i * self.sblk
                    coffset = offset

                inds[0, c : c + chunk] = (
                    torch.repeat_interleave(
                        torch.arange(
                            0, self.sblk, dtype=torch.int64, device=self.device
                        ).unsqueeze(-1),
                        self.sblk,
                        -1,
                    ).flatten()
                    + roffset
                )
                inds[1, c : c + chunk] = (
                    torch.repeat_interleave(
                        torch.arange(
                            0, self.sblk, dtype=torch.int64, device=self.device
                        ).unsqueeze(0),
                        self.sblk,
                        0,
                    ).flatten()
                    + coffset
                )

                c += chunk

        return torch.sparse_coo_tensor(
            inds,
            data,
            dtype=self.dtype,
            device=self.device,
            size=(self.n, self.n, self.sbat),
        ).coalesce()

    def to_unrolled_csr(self):
        """
        Return a list of CSR tensors with length equal to the batch size

        """
        coo = self.to_batched_coo()
        return [
            torch.sparse_coo_tensor(coo.indices(), coo.values()[:, i]).to_sparse_csr()
            for i in range(self.sbat)
        ]
