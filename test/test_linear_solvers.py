import unittest

import torch

from pyoptmat import solvers

torch.set_default_tensor_type(torch.DoubleTensor)

class CommonLinearSolver:
    def test_random(self):
        A = torch.rand((self.nbatch, self.nsize, self.nsize))/10.0 + torch.diag_embed(torch.ones((self.nbatch,self.nsize)))
        b = torch.rand((self.nbatch, self.nsize))

        x_exact = torch.linalg.solve(A,b)
        x_trial = self.solve_fn(A, b)
        
        self.assertTrue(torch.allclose(x_exact, x_trial))

class TestLUSolve(unittest.TestCase, CommonLinearSolver):
    def setUp(self):
        self.nsize = 5
        self.nbatch = 10

        self.solve_fn = solvers.lu_linear_solve

class TestDiagonalSolve(unittest.TestCase, CommonLinearSolver):
    def setUp(self):
        self.nsize = 1
        self.nbatch = 10

        self.solve_fn = solvers.diagonal_linear_solve

class TestJacobiSolve(unittest.TestCase, CommonLinearSolver):
    def setUp(self):
        self.nsize = 5
        self.nbatch = 10

        self.solve_fn = solvers.jacobi_linear_solve
