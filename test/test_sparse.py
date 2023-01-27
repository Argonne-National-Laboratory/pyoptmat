from pyoptmat import spsolve

import torch

import unittest

torch.set_default_tensor_type(torch.DoubleTensor)

class TestBasicSparseSetup(unittest.TestCase):
    def setUp(self):
        self.sblk = 4
        self.nblk = 3
        self.sbatch = 4

        self.u = torch.zeros(self.nblk-2, self.sbatch, self.sblk, self.sblk)
        self.d = torch.zeros(self.nblk, self.sbatch, self.sblk, self.sblk)
        self.l = torch.zeros(self.nblk-1, self.sbatch, self.sblk, self.sblk)

        for i in range(self.nblk-2):
            self.u[i] = (i + 2) * 1.0
        for i in range(self.nblk):
            self.d[i] = 2.0*i-1.0
        for i in range(self.nblk-1):
            self.l[i] = -(i + 1) * 1.0

        self.sp = spsolve.SquareBatchedBlockDiagonalMatrix([self.d, self.l, self.u],
                                                           [0, -1, 2])

    def test_coo(self):
        coo = self.sp.to_batched_coo()
        d = coo.to_dense().movedim(-1,0)
        od = self.sp.to_dense()

        self.assertTrue(torch.allclose(d, od))

    def test_csr(self):
        csr_list = self.sp.to_unrolled_csr()
        od = self.sp.to_dense()

        for i in range(self.sbatch):
            #print(csr_list[i].crow_indices())
            #print(csr_list[i].col_indices())
            #print(csr_list[i].values())
            self.assertTrue(torch.allclose(csr_list[i].to_dense(), od[i]))
