import unittest

import torch

from pyoptmat import solvers

torch.set_default_dtype(torch.float64)


class LinearSolverBase:
    def test_correct(self):
        A = torch.rand((self.nbatch, self.N, self.N))
        b = torch.rand((self.nbatch, self.N))

        x = self.method(A, b)

        xp = torch.linalg.solve(A, b)

        self.assertTrue(torch.allclose(x, xp))


class TestLU(unittest.TestCase, LinearSolverBase):
    def setUp(self):
        self.nbatch = 10
        self.N = 8

        self.method = solvers.lu_linear_solve


class TestGMRES(unittest.TestCase, LinearSolverBase):
    def setUp(self):
        self.nbatch = 20
        self.N = 25

        self.method = solvers.gmres


class TestGMRESDiag(unittest.TestCase, LinearSolverBase):
    def setUp(self):
        self.nbatch = 20
        self.N = 25

        self.method = solvers.jacobi_gmres


class TestGMRESLU(unittest.TestCase, LinearSolverBase):
    def setUp(self):
        self.nbatch = 20
        self.N = 25

        self.method = solvers.lu_gmres
