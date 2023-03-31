import unittest

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn

from pyoptmat import models, flowrules, utility, hardening, damage
from pyoptmat.temperature import ConstantParameter as CP

torch.set_default_dtype(torch.float64)


class CommonModel:
    def test_derivs_strain(self):
        strain_rates = torch.cat(
            (
                torch.zeros(1, self.strains.shape[1]),
                (self.strains[1:] - self.strains[:-1])
                / (self.times[1:] - self.times[:-1]),
            )
        )
        strain_rates[torch.isnan(strain_rates)] = 0

        #print(self.times.shape)
        #print(strain_rates.shape)
        #print(self.t.shape)

        erate_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            self.times, strain_rates
        )
        temperature_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            self.times, self.temperatures
        )

        use = models.StrainBasedModel(
            self.model, erate_interpolator, temperature_interpolator
        )
        v, dv = use.forward(self.t, self.state_strain)
        ddv = utility.batch_differentiate(
            lambda x: use.forward(self.t, x)[0], self.state_strain
        )

        self.assertTrue(np.allclose(dv, ddv, rtol=1e-4, atol=1e-4))

    def test_derivs_stress(self):
        stress_rates = torch.cat(
            (
                torch.zeros(1, self.stresses.shape[1]),
                (self.stresses[1:] - self.stresses[:-1])
                / (self.times[1:] - self.times[:-1]),
            )
        )
        stress_rates[torch.isnan(stress_rates)] = 0

        stress_rate_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            self.times, stress_rates
        )
        stress_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            self.times, self.stresses
        )
        temperature_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            self.times, self.temperatures
        )

        use = models.StressBasedModel(
            self.model,
            stress_rate_interpolator,
            stress_interpolator,
            temperature_interpolator,
        )

        v, dv = use.forward(self.t, self.state_stress)
        ddv = utility.batch_differentiate(
            lambda x: use.forward(self.t, x)[0], self.state_stress
        )

        self.assertTrue(np.allclose(dv, ddv, rtol=1e-4, atol=1e-4))

    def test_partial_state(self):
        strain_rates = torch.cat(
            (
                torch.zeros(1, self.strains.shape[1]),
                (self.strains[1:] - self.strains[:-1])
                / (self.times[1:] - self.times[:-1]),
            )
        )
        strain_rates[torch.isnan(strain_rates)] = 0

        t = self.times[self.step]
        T = self.temperatures[self.step]
        erate = strain_rates[self.step]

        _, exact, _, _ = self.model.forward(t, self.state_strain, erate, T)
        numer = utility.batch_differentiate(
            lambda y: self.model.forward(t, y, erate, T)[0], self.state_strain
        )

        self.assertTrue(torch.allclose(exact, numer, atol=1e-4, rtol=1e-4))

    def test_partial_erate(self):
        strain_rates = torch.cat(
            (
                torch.zeros(1, self.strains.shape[1]),
                (self.strains[1:] - self.strains[:-1])
                / (self.times[1:] - self.times[:-1]),
            )
        )
        strain_rates[torch.isnan(strain_rates)] = 0

        t = self.times[self.step]
        T = self.temperatures[self.step]
        erate = strain_rates[self.step]

        _, _, exact, _ = self.model.forward(t, self.state_strain, erate, T)
        numer = utility.batch_differentiate(
            lambda y: self.model.forward(t, self.state_strain, y, T)[0], erate
        ).squeeze(-1)

        self.assertTrue(torch.allclose(exact, numer, atol=1e-4, rtol=1e-4))


class CommonModelBatchBatch:
    nextra = 6

    def expand(self, T):
        return T.unsqueeze(0).expand((CommonModelBatchBatch.nextra,) + T.shape)

    def test_derivs_strain_bb(self):
        strain_rates = torch.cat(
            (
                torch.zeros(1, self.strains.shape[1]),
                (self.strains[1:] - self.strains[:-1])
                / (self.times[1:] - self.times[:-1]),
            )
        )
        strain_rates[torch.isnan(strain_rates)] = 0

        erate_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            self.times, strain_rates
        )
        temperature_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            self.times, self.temperatures
        )

        use = models.StrainBasedModel(
            self.model, erate_interpolator, temperature_interpolator
        )
        v, dv = use.forward(self.expand(self.t), self.expand(self.state_strain))
        ddv = utility.batch_differentiate(
            lambda x: use.forward(self.expand(self.t), x)[0],
            self.expand(self.state_strain),
            nbatch_dim=2,
        )

        self.assertTrue(np.allclose(dv, ddv, rtol=1e-4, atol=1e-4))

    def test_derivs_stress_bb(self):
        stress_rates = torch.cat(
            (
                torch.zeros(1, self.stresses.shape[1]),
                (self.stresses[1:] - self.stresses[:-1])
                / (self.times[1:] - self.times[:-1]),
            )
        )
        stress_rates[torch.isnan(stress_rates)] = 0

        stress_rate_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            self.times, stress_rates
        )
        stress_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            self.times, self.stresses
        )
        temperature_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            self.times, self.temperatures
        )

        use = models.StressBasedModel(
            self.model,
            stress_rate_interpolator,
            stress_interpolator,
            temperature_interpolator,
        )

        v, dv = use.forward(self.expand(self.t), self.expand(self.state_stress))
        ddv = utility.batch_differentiate(
            lambda x: use.forward(self.expand(self.t), x)[0],
            self.expand(self.state_stress),
            nbatch_dim=2,
        )

        self.assertTrue(np.allclose(dv, ddv, rtol=1e-4, atol=1e-4))

    def test_partial_state_bb(self):
        strain_rates = torch.cat(
            (
                torch.zeros(1, self.strains.shape[1]),
                (self.strains[1:] - self.strains[:-1])
                / (self.times[1:] - self.times[:-1]),
            )
        )
        strain_rates[torch.isnan(strain_rates)] = 0

        t = self.times[self.step]
        T = self.temperatures[self.step]
        erate = strain_rates[self.step]

        _, exact, _, _ = self.model.forward(
            self.expand(t),
            self.expand(self.state_strain),
            self.expand(erate),
            self.expand(T),
        )
        numer = utility.batch_differentiate(
            lambda y: self.model.forward(
                self.expand(t), y, self.expand(erate), self.expand(T)
            )[0],
            self.expand(self.state_strain),
            nbatch_dim=2,
        )

        self.assertTrue(torch.allclose(exact, numer, atol=1e-4, rtol=1e-4))

    def test_partial_erate_bb(self):
        strain_rates = torch.cat(
            (
                torch.zeros(1, self.strains.shape[1]),
                (self.strains[1:] - self.strains[:-1])
                / (self.times[1:] - self.times[:-1]),
            )
        )
        strain_rates[torch.isnan(strain_rates)] = 0

        t = self.times[self.step]
        T = self.temperatures[self.step]
        erate = strain_rates[self.step]

        _, _, exact, _ = self.model.forward(
            self.expand(t),
            self.expand(self.state_strain),
            self.expand(erate),
            self.expand(T),
        )
        numer = utility.batch_differentiate(
            lambda y: self.model.forward(
                self.expand(t), self.expand(self.state_strain), y, self.expand(T)
            )[0],
            self.expand(erate),
            nbatch_dim=2,
        )

        self.assertTrue(torch.allclose(exact, numer, atol=1e-4, rtol=1e-4))


class TestPerfectViscoplasticity(unittest.TestCase, CommonModel, CommonModelBatchBatch):
    def setUp(self):
        self.E = torch.tensor(100000.0)
        self.n = torch.tensor(5.2)
        self.eta = torch.tensor(110.0)

        self.times = torch.transpose(
            torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
        )
        self.strains = (
            torch.transpose(
                torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
            )
            / 10.0
        )
        self.temperatures = torch.zeros_like(self.strains)
        self.stresses = (
            torch.transpose(
                torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
            )
            * 0
        )

        self.state_strain = torch.tensor([[90.0], [100.0], [101.0]])
        self.state_stress = torch.tensor([[0.0], [0.0], [0.0]])
        self.t = self.times[2]

        self.flowrule = flowrules.PerfectViscoplasticity(CP(self.n), CP(self.eta))
        self.model = models.InelasticModel(CP(self.E), self.flowrule)
        self.step = 2


class TestIsoKinViscoplasticity(unittest.TestCase, CommonModel, CommonModelBatchBatch):
    def setUp(self):
        self.E = torch.tensor(100000.0)
        self.n = torch.tensor(5.2)
        self.eta = torch.tensor(110.0)
        self.s0 = torch.tensor(0.0)

        self.R = torch.tensor(101.0)
        self.d = torch.tensor(1.3)
        self.iso = hardening.VoceIsotropicHardeningModel(CP(self.R), CP(self.d))

        self.C = torch.tensor(12000.0)
        self.g = torch.tensor(10.1)
        self.kin = hardening.FAKinematicHardeningModel(CP(self.C), CP(self.g))

        self.flowrule = flowrules.IsoKinViscoplasticity(
            CP(self.n), CP(self.eta), CP(self.s0), self.iso, self.kin
        )
        self.model = models.InelasticModel(CP(self.E), self.flowrule)

        self.times = torch.transpose(
            torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
        )
        self.strains = torch.transpose(
            torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
        )
        self.temperatures = torch.zeros_like(self.times)
        self.stresses = (
            torch.transpose(
                torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
            )
            * 200
        )

        self.state_strain = (
            torch.tensor(
                [[90.0, 30.0, 10.0, 0], [100.0, 10.0, 15.0, 0], [101.0, 50.0, 60.0, 0]]
            )
            / 3
        )
        self.state_stress = (
            torch.tensor(
                [[0.05, 30.0, 10.0, 0], [0.07, 10.0, 15.0, 0], [0.08, 50.0, 60.0, 0]]
            )
            / 3
        )

        self.t = self.times[2]
        self.step = 2


class TestIsoKinViscoplasticityRecovery(
    unittest.TestCase, CommonModel, CommonModelBatchBatch
):
    def setUp(self):
        self.E = torch.tensor(100000.0)
        self.n = torch.tensor(5.2)
        self.eta = torch.tensor(110.0)
        self.s0 = torch.tensor(0.0)

        self.tau0 = torch.tensor(101.0)
        self.theta0 = torch.tensor(1000.0)
        self.R0 = torch.tensor(0.0)
        self.r1 = torch.tensor(1.0e-6)
        self.r2 = torch.tensor(2.0)
        self.iso = hardening.Theta0RecoveryVoceIsotropicHardeningModel(
            CP(self.tau0), CP(self.theta0), CP(self.R0), CP(self.r1), CP(self.r2)
        )

        self.C = torch.tensor(12000.0)
        self.g = torch.tensor(10.1)
        self.kin = hardening.FAKinematicHardeningModel(CP(self.C), CP(self.g))

        self.flowrule = flowrules.IsoKinViscoplasticity(
            CP(self.n), CP(self.eta), CP(self.s0), self.iso, self.kin
        )
        self.model = models.InelasticModel(CP(self.E), self.flowrule)

        self.times = torch.transpose(
            torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
        )
        self.strains = torch.transpose(
            torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
        )
        self.temperatures = torch.zeros_like(self.times)
        self.stresses = (
            torch.transpose(
                torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
            )
            * 200
        )

        self.state_strain = (
            torch.tensor(
                [[90.0, 30.0, 10.0, 0], [100.0, 10.0, 15.0, 0], [101.0, 50.0, 60.0, 0]]
            )
            / 3
        )
        self.state_stress = (
            torch.tensor(
                [[0.05, 30.0, 10.0, 0], [0.07, 10.0, 15.0, 0], [0.08, 50.0, 60.0, 0]]
            )
            / 3
        )

        self.t = self.times[2]
        self.step = 2


class TestDamage(unittest.TestCase, CommonModel, CommonModelBatchBatch):
    def setUp(self):
        self.E = torch.tensor(100000.0)
        self.n = torch.tensor(5.2)
        self.eta = torch.tensor(110.0)
        self.s0 = torch.tensor(0.0)

        self.R = torch.tensor(101.0)
        self.d = torch.tensor(1.3)
        self.iso = hardening.VoceIsotropicHardeningModel(CP(self.R), CP(self.d))

        self.C = torch.tensor(1200.0)
        self.g = torch.tensor(10.1)
        self.kin = hardening.FAKinematicHardeningModel(CP(self.C), CP(self.g))

        self.A = torch.tensor(3000.0)
        self.xi = torch.tensor(6.5)
        self.phi = torch.tensor(1.7)
        self.dmodel = damage.HayhurstLeckie(CP(self.A), CP(self.xi), CP(self.phi))

        self.flowrule = flowrules.IsoKinViscoplasticity(
            CP(self.n), CP(self.eta), CP(self.s0), self.iso, self.kin
        )
        self.model = models.InelasticModel(
            CP(self.E), self.flowrule, dmodel=self.dmodel
        )

        self.times = torch.transpose(
            torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
        )
        self.strains = torch.transpose(
            torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
        )
        self.temperatures = torch.zeros_like(self.strains)
        self.stresses = (
            torch.transpose(
                torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
            )
            * 200
        )

        self.state_strain = torch.tensor(
            [[90.0, 30.0, 10.0, 0.05], [100.0, 10.0, 15.0, 0.1], [20, -10.0, -10, 0.2]]
        )
        self.state_stress = torch.tensor(
            [[0.1, 30.0, 10.0, 0.05], [0.11, 10.0, 15.0, 0.1], [0.12, -10.0, -10, 0.2]]
        )

        self.t = self.times[2]
        self.step = 2


class TestAll(unittest.TestCase, CommonModel, CommonModelBatchBatch):
    def setUp(self):
        self.E = torch.tensor(100000.0)
        self.n = torch.tensor(5.2)
        self.eta = torch.tensor(110.0)
        self.s0 = torch.tensor(0.0)

        self.R = torch.tensor(101.0)
        self.d = torch.tensor(1.3)
        self.iso = hardening.VoceIsotropicHardeningModel(CP(self.R), CP(self.d))

        self.C = torch.tensor([1200.0, 200.0, 10.0])
        self.g = torch.tensor([10.1, 100.0, 50.0])
        self.kin = hardening.ChabocheHardeningModel(CP(self.C), CP(self.g))

        self.A = torch.tensor(3000.0)
        self.xi = torch.tensor(6.5)
        self.phi = torch.tensor(1.7)
        self.dmodel = damage.HayhurstLeckie(CP(self.A), CP(self.xi), CP(self.phi))

        self.flowrule = flowrules.IsoKinViscoplasticity(
            CP(self.n), CP(self.eta), CP(self.s0), self.iso, self.kin
        )
        self.model = models.InelasticModel(
            CP(self.E), self.flowrule, dmodel=self.dmodel
        )

        self.times = torch.transpose(
            torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
        )
        self.strains = torch.transpose(
            torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
        )
        self.temperatures = torch.zeros_like(self.strains)
        self.stresses = (
            torch.transpose(
                torch.tensor(np.array([np.linspace(0, 1, 4) for i in range(3)])), 1, 0
            )
            * 200
        )

        self.state_strain = torch.tensor(
            [
                [90.0, 30.0, 10.0, 10.0, -10.0, 0.2],
                [100.0, 10.0, 15.0, 5.0, -10.0, 0.3],
                [101.0, 50.0, 60.0, -50.0, 10.0, 0.4],
            ]
        )
        self.state_stress = torch.tensor(
            [
                [0.05, 30.0, 10.0, 10.0, -10.0, 0.2],
                [0.08, 10.0, 15.0, 5.0, -10.0, 0.3],
                [0.07, 50.0, 60.0, -50.0, 10.0, 0.4],
            ]
        )

        self.t = self.times[2]
        self.step = 2
