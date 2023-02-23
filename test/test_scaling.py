import unittest

import torch

from pyoptmat import scaling


class ScalingTests:
    def test_forward_back(self):
        actual = self.f.scale(self.x)
        back = self.f.unscale(actual)

        self.assertTrue(torch.allclose(self.x, back))


class TestBoundedScalingFunction(unittest.TestCase, ScalingTests):
    def setUp(self):
        self.l = torch.tensor([2.0, -1.0])
        self.u = torch.tensor([12.0, 2.0])

        self.x = torch.tensor([0.6, 0.5])

        self.f = scaling.BoundedScalingFunction(self.l, self.u)


class TestLogBoundedScalingFunction(unittest.TestCase, ScalingTests):
    def setUp(self):
        self.l = torch.tensor([2.0, 0.5])
        self.u = torch.tensor([12.0, 2.0])

        self.x = torch.tensor([0.6, 0.5])

        self.f = scaling.LogBoundedScalingFunction(self.l, self.u)
