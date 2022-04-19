import unittest

import numpy as np

import torch

from pyoptmat import hardening, utility
from pyoptmat.temperature import ConstantParameter as CP

torch.set_default_dtype(torch.float64)


class HardeningBase:
    def test_dvalue(self):
        exact = self.model.dvalue(self.h)
        numer = utility.new_differentiate(lambda x: self.model.value(x), self.h)

        self.assertTrue(np.allclose(exact, numer, rtol=1.0e-4))

    def test_dstress(self):
        exact = self.model.dhistory_rate_dstress(
            self.s, self.h, self.t, self.ep, self.T
        )
        numer = utility.new_differentiate(
            lambda x: self.model.history_rate(x, self.h, self.t, self.ep, self.T),
            self.s,
        )

        self.assertTrue(np.allclose(exact, numer[:, :, 0], rtol=1.0e-4))

    def test_dhistory(self):
        exact = self.model.dhistory_rate_dhistory(
            self.s, self.h, self.t, self.ep, self.T
        )
        numer = utility.new_differentiate(
            lambda x: self.model.history_rate(self.s, x, self.t, self.ep, self.T),
            self.h,
        )

        print(exact.shape)
        print(numer.shape)

        self.assertTrue(np.allclose(exact, numer, rtol=1.0e-3))

    def test_derate(self):
        exact = self.model.dhistory_rate_derate(self.s, self.h, self.t, self.ep, self.T)
        numer = utility.new_differentiate(
            lambda x: self.model.history_rate(self.s, self.h, self.t, x, self.T),
            self.ep,
        )

        self.assertTrue(np.allclose(exact, numer, rtol=1.0e-4))


class TestVoceIsotropicHardening(unittest.TestCase, HardeningBase):
    def setUp(self):
        self.R = torch.tensor(100.0)
        self.d = torch.tensor(1.2)
        self.model = hardening.VoceIsotropicHardeningModel(CP(self.R), CP(self.d))

        self.nbatch = 10

        self.s = torch.linspace(90, 100, self.nbatch)
        self.h = torch.reshape(torch.linspace(50, 110, self.nbatch), (self.nbatch, 1))
        self.t = torch.ones(self.nbatch)
        self.ep = torch.linspace(0.1, 0.2, self.nbatch)
        self.T = torch.zeros_like(self.t)


class TestVoceIsotropicThetaHardening(unittest.TestCase, HardeningBase):
    def setUp(self):
        self.tau = torch.tensor(100.0)
        self.theta = torch.tensor(12.0)
        self.model = hardening.Theta0VoceIsotropicHardeningModel(
            CP(self.tau), CP(self.theta)
        )

        self.nbatch = 10

        self.s = torch.linspace(90, 100, self.nbatch)
        self.h = torch.reshape(torch.linspace(50, 110, self.nbatch), (self.nbatch, 1))
        self.t = torch.ones(self.nbatch)
        self.ep = torch.linspace(0.1, 0.2, self.nbatch)
        self.T = torch.zeros_like(self.t)


class TestVoceIsotropicThetaReceoveryHardening(unittest.TestCase, HardeningBase):
    def setUp(self):
        self.tau = torch.tensor(100.0)
        self.theta = torch.tensor(12.0)
        self.r1 = 0.1
        self.r2 = 1.2
        self.R0 = 10.0
        self.model = hardening.Theta0RecoveryVoceIsotropicHardeningModel(
            CP(self.tau), CP(self.theta), CP(self.R0), CP(self.r1), CP(self.r2)
        )

        self.nbatch = 10

        self.s = torch.linspace(90, 100, self.nbatch)
        self.h = torch.reshape(torch.linspace(50, 110, self.nbatch), (self.nbatch, 1))
        self.t = torch.ones(self.nbatch)
        self.ep = torch.linspace(0.1, 0.2, self.nbatch)
        self.T = torch.zeros_like(self.t)


class TestFAKinematicHardening(unittest.TestCase, HardeningBase):
    def setUp(self):
        self.C = torch.tensor(100.0)
        self.g = torch.tensor(1.2)
        self.model = hardening.FAKinematicHardeningModel(CP(self.C), CP(self.g))

        self.nbatch = 10

        self.s = torch.linspace(90, 100, self.nbatch)
        self.h = torch.reshape(torch.linspace(50, 110, self.nbatch), (self.nbatch, 1))
        self.t = torch.ones(self.nbatch)
        self.ep = torch.linspace(0.1, 0.2, self.nbatch)
        self.T = torch.zeros_like(self.t)


class TestFAKinematicHardeningRecovery(unittest.TestCase, HardeningBase):
    def setUp(self):
        self.C = torch.tensor(100.0)
        self.g = torch.tensor(1.2)
        self.b = torch.tensor(5.0e-4)
        self.r = torch.tensor(3.0)
        self.model = hardening.FAKinematicHardeningModel(
            CP(self.C), CP(self.g), CP(self.b), CP(self.r)
        )

        self.nbatch = 10

        self.s = torch.linspace(90, 100, self.nbatch)
        self.h = torch.reshape(torch.linspace(50, 110, self.nbatch), (self.nbatch, 1))
        self.t = torch.ones(self.nbatch)
        self.ep = torch.linspace(0.1, 0.2, self.nbatch)
        self.T = torch.zeros_like(self.t)


class TestSuperimposedKinematicHardening(unittest.TestCase, HardeningBase):
    def setUp(self):
        self.C1 = torch.tensor(100.0)
        self.g1 = torch.tensor(1.2)
        self.model1 = hardening.FAKinematicHardeningModel(CP(self.C1), CP(self.g1))

        self.C2 = torch.tensor(12.0)
        self.g2 = torch.tensor(1.5)
        self.model2 = hardening.FAKinematicHardeningModel(CP(self.C2), CP(self.g2))

        self.model = hardening.SuperimposedKinematicHardening(
            [self.model1, self.model2]
        )

        self.nbatch = 10

        self.s = torch.linspace(90, 100, self.nbatch)
        self.h = torch.reshape(
            torch.linspace(50, 110, 2 * self.nbatch), (self.nbatch, 2)
        )
        self.t = torch.ones(self.nbatch)
        self.ep = torch.linspace(0.1, 0.2, self.nbatch)
        self.T = torch.zeros_like(self.t)

    def test_correct_sum(self):
        should = torch.sum(self.h, 1)
        model = self.model.value(self.h)

        self.assertTrue(np.allclose(should.numpy(), model.numpy()))

    def test_correct_rate(self):
        rates = self.model.history_rate(self.s, self.h, self.t, self.ep, self.T)

        self.assertTrue(
            np.allclose(
                self.model1.history_rate(
                    self.s, self.h[:, :1], self.t, self.ep, self.T
                ),
                rates[:, :1],
            )
        )
        self.assertTrue(
            np.allclose(
                self.model2.history_rate(
                    self.s, self.h[:, 1:2], self.t, self.ep, self.T
                ),
                rates[:, 1:2],
            )
        )


class TestChabocheKinematicHardening(unittest.TestCase, HardeningBase):
    def setUp(self):
        self.C = torch.tensor([100.0, 1000, 1500])
        self.g = torch.tensor([1.2, 100, 50])
        self.model = hardening.ChabocheHardeningModel(CP(self.C), CP(self.g))

        self.nbatch = 10

        self.s = torch.linspace(90, 100, self.nbatch)
        self.h = torch.reshape(
            torch.linspace(50, 110, self.nbatch * len(self.C)),
            (self.nbatch, len(self.C)),
        )
        self.t = torch.ones(self.nbatch)
        self.ep = torch.linspace(0.1, 0.2, self.nbatch)
        self.T = torch.zeros_like(self.t)


class TestChabocheKinematicHardeningRecovery(unittest.TestCase, HardeningBase):
    def setUp(self):
        self.C = torch.tensor([100.0, 1000, 1500])
        self.g = torch.tensor([1.2, 100, 50])
        self.b = torch.tensor([5e-4, 4e-4, 2e-4])
        self.r = torch.tensor([3.0, 3.2, 3.5])
        self.model = hardening.ChabocheHardeningModelRecovery(
            CP(self.C), CP(self.g), CP(self.b), CP(self.r)
        )

        self.nbatch = 10

        self.s = torch.linspace(90, 100, self.nbatch)
        self.h = torch.reshape(
            torch.linspace(50, 110, self.nbatch * len(self.C)),
            (self.nbatch, len(self.C)),
        )
        self.t = torch.ones(self.nbatch)
        self.ep = torch.linspace(0.1, 0.2, self.nbatch)
        self.T = torch.zeros_like(self.t)
