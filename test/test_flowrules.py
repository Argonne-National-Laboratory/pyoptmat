import unittest

import numpy as np

import torch

from pyoptmat import flowrules, hardening, utility
from pyoptmat.temperature import ConstantParameter as CP

torch.set_default_dtype(torch.float64)


class CommonFlowRule:
    def test_flow_rate(self):
        exact = self.model.flow_rate(self.s, self.h, self.t, self.T, self.erate)[1]
        numer = utility.differentiate(
            lambda x: self.model.flow_rate(x, self.h, self.t, self.T, self.erate)[0], self.s
        )

        self.assertTrue(np.allclose(exact, numer, rtol=1.0e-4))

    def test_history_rate(self):
        if self.skip:
            return

        test, exact = self.model.history_rate(self.s, self.h, self.t, self.T, self.erate)
        numer = utility.new_differentiate(
            lambda x: self.model.history_rate(self.s, x, self.t, self.T, self.erate)[0], self.h
        )

        self.assertTrue(np.allclose(exact, numer, rtol=1.0e-4))

    def test_flow_history(self):
        if self.skip:
            return

        exact = self.model.dflow_dhist(self.s, self.h, self.t, self.T, self.erate)
        numer = utility.new_differentiate(
            lambda x: self.model.flow_rate(self.s, x, self.t, self.T, self.erate)[0], self.h
        )

        self.assertTrue(np.allclose(exact, numer, rtol=1.0e-4))

    def test_history_stress(self):
        if self.skip:
            return

        exact = self.model.dhist_dstress(self.s, self.h, self.t, self.T, self.erate)
        numer = utility.new_differentiate(
            lambda x: self.model.history_rate(x, self.h, self.t, self.T, self.erate)[0], self.s
        )[..., 0]

        self.assertTrue(np.allclose(exact, numer, rtol=1.0e-4))

    def test_flow_erate(self):
        exact = self.model.dflow_derate(self.s, self.h, self.t, self.T, self.erate)
        numer = utility.new_differentiate(
            lambda x: self.model.flow_rate(self.s, self.h, self.t, self.T, x)[0], self.erate
        )

        self.assertTrue(np.allclose(exact, numer, rtol=1.0e-4))

    def test_history_erate(self):
        if self.skip:
            return

        exact = self.model.dhist_derate(self.s, self.h, self.t, self.T, self.erate)
        numer = utility.new_differentiate(
            lambda x: self.model.history_rate(self.s, self.h, self.t, self.T, x)[0], self.erate
        )[..., 0]

        self.assertTrue(np.allclose(exact, numer, rtol=1.0e-4))


class TestPerfectViscoplasticity(unittest.TestCase, CommonFlowRule):
    def setUp(self):
        self.n = torch.tensor(5.0)
        self.eta = torch.tensor(100.0)

        self.nbatch = 10

        self.model = flowrules.PerfectViscoplasticity(CP(self.n), CP(self.eta))

        self.s = torch.linspace(90, 100, self.nbatch)
        self.h = torch.reshape(torch.linspace(50, 110, self.nbatch), (self.nbatch, 1))
        self.t = torch.ones(self.nbatch)
        self.skip = True
        self.T = torch.zeros_like(self.t)
        self.erate = torch.linspace(1e-2,1e-3,self.nbatch)


class TestIsoKinViscoplasticity(unittest.TestCase, CommonFlowRule):
    def setUp(self):
        self.n = torch.tensor(5.2)
        self.eta = torch.tensor(110.0)
        self.s0 = torch.tensor(11.0)

        self.nbatch = 10

        self.R = torch.tensor(101.0)
        self.d = torch.tensor(1.3)
        self.iso = hardening.VoceIsotropicHardeningModel(CP(self.R), CP(self.d))

        self.C = torch.tensor(1200.0)
        self.g = torch.tensor(10.1)
        self.kin = hardening.FAKinematicHardeningModel(CP(self.C), CP(self.g))

        self.model = flowrules.IsoKinViscoplasticity(
            CP(self.n), CP(self.eta), CP(self.s0), self.iso, self.kin
        )

        self.s = torch.linspace(150, 200, self.nbatch)
        self.h = torch.reshape(
            torch.tensor(
                np.array(
                    [
                        np.linspace(51, 110, self.nbatch),
                        np.linspace(-100, 210, self.nbatch)[::-1],
                    ]
                )
            ).T,
            (self.nbatch, 2),
        )

        self.t = torch.ones(self.nbatch)
        self.T = torch.zeros_like(self.t)
        self.erate = torch.linspace(1e-2,1e-3,self.nbatch)

        self.skip = False

    def test_kin(self):
        def fn(i):
            hp = self.h.clone()
            hp[:, 1] = i
            return self.model.flow_rate(self.s, hp, self.t, self.T, self.erate)[0]

        i1 = utility.differentiate(fn, self.h[:, 1])
        i2 = self.model.dflow_dkin(self.s, self.h, self.t, self.T, self.erate)
        self.assertTrue(np.allclose(i1, i2, rtol=1.0e-4))

    def test_iso(self):
        def fn(i):
            hp = self.h.clone()
            hp[:, 0] = i
            return self.model.flow_rate(self.s, hp, self.t, self.T, self.erate)[0]

        i1 = utility.differentiate(fn, self.h[:, 0])
        i2 = self.model.dflow_diso(self.s, self.h, self.t, self.T, self.erate)

        self.assertTrue(np.allclose(i1, i2, rtol=1.0e-4))


class TestSuperimposedFlowRate(unittest.TestCase, CommonFlowRule):
    def setUp(self):
        self.n1 = torch.tensor(5.2)
        self.eta1 = torch.tensor(110.0)
        self.s01 = torch.tensor(11.0)

        self.nbatch = 10

        self.R1 = torch.tensor(101.0)
        self.d1 = torch.tensor(1.3)
        self.iso1 = hardening.VoceIsotropicHardeningModel(CP(self.R1), CP(self.d1))

        self.C1 = torch.tensor(1200.0)
        self.g1 = torch.tensor(10.1)
        self.kin1 = hardening.FAKinematicHardeningModel(CP(self.C1), CP(self.g1))

        self.model1 = flowrules.IsoKinViscoplasticity(
            CP(self.n1), CP(self.eta1), CP(self.s01), self.iso1, self.kin1
        )

        self.n2 = torch.tensor(4.1)
        self.eta2 = torch.tensor(100.0)
        self.s02 = torch.tensor(1.0)

        self.R2 = torch.tensor(250.0)
        self.d2 = torch.tensor(20.1)
        self.iso2 = hardening.VoceIsotropicHardeningModel(CP(self.R2), CP(self.d2))

        self.C2 = torch.tensor(120.0)
        self.g2 = torch.tensor(11.1)
        self.kin2 = hardening.FAKinematicHardeningModel(CP(self.C2), CP(self.g2))

        self.model2 = flowrules.IsoKinViscoplasticity(
            CP(self.n2), CP(self.eta2), CP(self.s02), self.iso2, self.kin2
        )

        self.model = flowrules.SuperimposedFlowRule([self.model1, self.model2])

        self.s = torch.linspace(150, 200, self.nbatch)
        self.h = torch.reshape(
            torch.tensor(
                np.array(
                    [
                        np.linspace(51, 110, self.nbatch * 2),
                        np.linspace(-100, 210, self.nbatch * 2)[::-1],
                    ]
                )
            ).T,
            (self.nbatch, 4),
        )

        self.t = torch.ones(self.nbatch)
        self.T = torch.zeros_like(self.t)
        self.erate = torch.linspace(1e-2,1e-3,self.nbatch)

        self.skip = False


class TestIsoKinChabocheViscoplasticity(unittest.TestCase, CommonFlowRule):
    def setUp(self):
        self.n = torch.tensor(5.2)
        self.eta = torch.tensor(110.0)
        self.s0 = torch.tensor(11.0)

        self.nbatch = 10

        self.R = torch.tensor(101.0)
        self.d = torch.tensor(1.3)
        self.iso = hardening.VoceIsotropicHardeningModel(CP(self.R), CP(self.d))

        self.C = torch.tensor([100.0, 1000, 1500])
        self.g = torch.tensor([1.2, 100, 50])
        self.kin = hardening.ChabocheHardeningModel(CP(self.C), CP(self.g))

        self.model = flowrules.IsoKinViscoplasticity(
            CP(self.n), CP(self.eta), CP(self.s0), self.iso, self.kin
        )

        self.s = torch.linspace(150, 200, self.nbatch)
        self.h = torch.reshape(
            torch.tensor(
                np.array(
                    [
                        np.linspace(51, 110, self.nbatch),
                        np.linspace(-10, 21, self.nbatch)[::-1],
                        np.linspace(0, 2, self.nbatch),
                        np.linspace(-2, 0, self.nbatch),
                    ]
                )
            ).T,
            (self.nbatch, 4),
        )
        self.t = torch.ones(self.nbatch)
        self.T = torch.zeros_like(self.t)
        self.erate = torch.linspace(1e-2,1e-3,self.nbatch)

        self.skip = False

    def test_kin(self):
        def fn(i):
            hp = self.h.clone()
            hp[:, 1] = i
            return self.model.flow_rate(self.s, hp, self.t, self.T, self.erate)[0]

        i1 = utility.differentiate(fn, self.h[:, 1])
        i2 = self.model.dflow_dkin(self.s, self.h, self.t, self.T, self.erate)
        self.assertTrue(np.allclose(i1, i2, rtol=1.0e-4))

    def test_iso(self):
        def fn(i):
            hp = self.h.clone()
            hp[:, 0] = i
            return self.model.flow_rate(self.s, hp, self.t, self.T, self.erate)[0]

        i1 = utility.differentiate(fn, self.h[:, 0])
        i2 = self.model.dflow_diso(self.s, self.h, self.t, self.T, self.erate)

        self.assertTrue(np.allclose(i1, i2, rtol=1.0e-4))
