import unittest

import numpy as np

import torch

from pyoptmat import damage, utility
from pyoptmat.temperature import ConstantParameter as CP

torch.set_default_dtype(torch.float64)


class DamageBase:
    def test_d_rate(self):
        exact = self.model.damage_rate(self.s, self.d, self.t, self.T, self.erate)[1]
        numer = utility.batch_differentiate(
            lambda x: self.model.damage_rate(self.s, x, self.t, self.T, self.erate)[0],
            self.d,
            nbatch_dim=self.bdim,
        )
        self.assertTrue(np.allclose(exact, numer))

    def test_d_stress(self):
        exact = self.model.d_damage_rate_d_s(self.s, self.d, self.t, self.T, self.erate)
        numer = utility.batch_differentiate(
            lambda x: self.model.damage_rate(x, self.d, self.t, self.T, self.erate)[0],
            self.s,
            nbatch_dim=self.bdim,
        )
        self.assertTrue(np.allclose(exact, numer))

    def test_d_erate(self):
        exact = self.model.d_damage_rate_d_e(self.s, self.d, self.t, self.T, self.erate)
        numer = utility.batch_differentiate(
            lambda x: self.model.damage_rate(self.s, self.d, self.t, self.T, x)[0],
            self.erate,
            nbatch_dim=self.bdim,
        )
        self.assertTrue(np.allclose(exact, numer))


def batch_transform(tensor, extra_batch):
    os = tensor.shape
    return tensor.unsqueeze(0).expand((extra_batch,) + os)


class TestNoDamage(unittest.TestCase, DamageBase):
    def setUp(self):
        self.model = damage.NoDamage()

        self.nbatch = 10
        self.bdim = 1

        self.s = torch.linspace(90, 100, self.nbatch)
        self.d = torch.linspace(0.1, 0.5, self.nbatch)
        self.t = torch.ones(self.nbatch)
        self.T = torch.zeros_like(self.t)
        self.erate = torch.linspace(1e-2, 1e-3, self.nbatch)

    def test_damage_rate(self):
        self.assertTrue(
            np.allclose(
                self.model.damage_rate(self.s, self.d, self.t, self.T, self.erate)[0],
                torch.zeros_like(self.s),
            )
        )


class TestNoDamageArbitraryBatch(unittest.TestCase, DamageBase):
    def setUp(self):
        self.model = damage.NoDamage()

        self.nbatch = 10
        self.extra_batch = 6
        self.bdim = 2

        self.s = batch_transform(torch.linspace(90, 100, self.nbatch), self.extra_batch)
        self.d = batch_transform(
            torch.linspace(0.1, 0.5, self.nbatch), self.extra_batch
        )
        self.t = batch_transform(torch.ones(self.nbatch), self.extra_batch)
        self.T = batch_transform(torch.zeros_like(self.t), self.extra_batch)
        self.erate = batch_transform(
            torch.linspace(1e-2, 1e-3, self.nbatch), self.extra_batch
        )

    def test_damage_rate(self):
        self.assertTrue(
            np.allclose(
                self.model.damage_rate(self.s, self.d, self.t, self.T, self.erate)[0],
                torch.zeros_like(self.s),
            )
        )


class TestHLDamage(unittest.TestCase, DamageBase):
    def setUp(self):
        self.A = 3000.0
        self.xi = 6.5
        self.phi = 1.7
        self.model = damage.HayhurstLeckie(CP(self.A), CP(self.xi), CP(self.phi))

        self.nbatch = 10
        self.bdim = 1

        self.s = torch.linspace(90, 100, self.nbatch)
        self.d = torch.linspace(0.1, 0.5, self.nbatch)
        self.t = torch.ones(self.nbatch)
        self.T = torch.zeros_like(self.t)
        self.erate = torch.linspace(1e-2, 1e-3, self.nbatch)

    def test_damage_rate(self):
        self.assertTrue(
            np.allclose(
                self.model.damage_rate(self.s, self.d, self.t, self.T, self.erate)[0],
                (self.s / self.A) ** (self.xi) * (1 - self.d) ** (self.xi - self.phi),
            )
        )


class TestHLDamageArbitraryBatch(unittest.TestCase, DamageBase):
    def setUp(self):
        self.A = 3000.0
        self.xi = 6.5
        self.phi = 1.7
        self.model = damage.HayhurstLeckie(CP(self.A), CP(self.xi), CP(self.phi))

        self.nbatch = 10
        self.extra_batch = 7
        self.bdim = 2

        self.s = batch_transform(torch.linspace(90, 100, self.nbatch), self.extra_batch)
        self.d = batch_transform(
            torch.linspace(0.1, 0.5, self.nbatch), self.extra_batch
        )
        self.t = batch_transform(torch.ones(self.nbatch), self.extra_batch)
        self.T = batch_transform(torch.zeros_like(self.t), self.extra_batch)
        self.erate = batch_transform(
            torch.linspace(1e-2, 1e-3, self.nbatch), self.extra_batch
        )

    def test_damage_rate(self):
        self.assertTrue(
            np.allclose(
                self.model.damage_rate(self.s, self.d, self.t, self.T, self.erate)[0],
                (self.s / self.A) ** (self.xi) * (1 - self.d) ** (self.xi - self.phi),
            )
        )
