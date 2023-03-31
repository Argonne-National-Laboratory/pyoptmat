import unittest

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn

from pyoptmat import ode, models, flowrules, hardening, utility, damage
from pyoptmat.temperature import ConstantParameter as CP

torch.set_default_tensor_type(torch.DoubleTensor)


def differ(mfn, p0, eps=1.0e-6):
    v0 = mfn(p0).numpy()

    puse = p0.numpy()

    result = np.zeros(puse.shape)

    for ind, val in np.ndenumerate(puse):
        dp = np.abs(val) * eps
        if dp < eps:
            dp = eps
        pcurr = np.copy(puse)
        pcurr[ind] += dp
        v1 = mfn(torch.tensor(pcurr)).numpy()
        result[ind] = (v1 - v0) / dp

    return result


def simple_diff(fn, p0):
    res = []
    for i in range(len(p0)):

        def mfn(pi):
            ps = [pp for pp in p0]
            ps[i] = pi
            return fn(ps)

        res.append(differ(mfn, p0[i]))

    return res


class CommonGradient:
    def test_gradient_strain(self):
        bmodel = self.model_fn([Variable(pi, requires_grad=True) for pi in self.p])
        res = torch.norm(
            bmodel.solve_strain(self.times, self.strains, self.temperatures)
        )
        res.backward()
        grad = self.extract_grad(bmodel)

        ngrad = simple_diff(
            lambda p: torch.norm(
                self.model_fn(p).solve_strain(
                    self.times, self.strains, self.temperatures
                )
            ),
            self.p,
        )

        for i, (p1, p2) in enumerate(zip(grad, ngrad)):
            print(i, p1, p2)
            self.assertTrue(np.allclose(p1, p2, rtol=1e-4))

    def test_gradient_stress(self):
        bmodel = self.model_fn([Variable(pi, requires_grad=True) for pi in self.p])
        res = torch.norm(
            bmodel.solve_stress(self.times, self.stresses, self.temperatures)
        )
        res.backward()
        grad = self.extract_grad(bmodel)

        ngrad = simple_diff(
            lambda p: torch.norm(
                self.model_fn(p).solve_stress(
                    self.times, self.stresses, self.temperatures
                )
            ),
            self.p,
        )

        # Skipping the first step helps with noise issues
        for i, (p1, p2) in enumerate(zip(grad[1:], ngrad[1:])):
            print(i, p1, p2)
            self.assertTrue(np.allclose(p1, p2, rtol=1e-4, atol=1e-7))


class TestPerfectViscoplasticity(unittest.TestCase, CommonGradient):
    def setUp(self):
        self.ntime = 10
        self.nbatch = 10

        self.E = torch.tensor(100000.0)
        self.n = torch.tensor(5.2)
        self.eta = torch.tensor(110.0)

        self.p = [self.E, self.n, self.eta]

        self.model_fn = lambda p: models.ModelIntegrator(
            models.InelasticModel(
                CP(p[0]), flowrules.PerfectViscoplasticity(CP(p[1]), CP(p[2]))
            ),
            use_adjoint=False,
        )

        self.extract_grad = lambda m: [
            m.model.E.pvalue.grad.numpy(),
            m.model.flowrule.n.pvalue.grad.numpy(),
            m.model.flowrule.eta.pvalue.grad.numpy(),
        ]

        self.times = torch.transpose(
            torch.tensor(
                np.array([np.linspace(0, 1, self.ntime) for i in range(self.nbatch)])
            ),
            1,
            0,
        )
        self.strains = torch.transpose(
            torch.tensor(
                np.array(
                    [np.linspace(0, 0.003, self.ntime) for i in range(self.nbatch)]
                )
            ),
            1,
            0,
        )
        self.stresses = torch.transpose(
            torch.tensor(
                np.array(
                    [np.linspace(0, 100.0, self.ntime) for i in range(self.nbatch)]
                )
            ),
            1,
            0,
        )
        self.temperatures = torch.zeros_like(self.strains)


class TestIsotropicOnly(unittest.TestCase, CommonGradient):
    def setUp(self):
        self.ntime = 10
        self.nbatch = 10

        self.E = torch.tensor(100000.0)
        self.n = torch.tensor(5.2)
        self.eta = torch.tensor(110.0)
        self.R = torch.tensor(100.0)
        self.d = torch.tensor(5.1)
        self.s0 = torch.tensor(10.0)

        self.p = [self.E, self.n, self.eta, self.s0, self.R, self.d]

        self.model_fn = lambda p: models.ModelIntegrator(
            models.InelasticModel(
                CP(p[0]),
                flowrules.IsoKinViscoplasticity(
                    CP(p[1]),
                    CP(p[2]),
                    CP(p[3]),
                    hardening.VoceIsotropicHardeningModel(CP(p[4]), CP(p[5])),
                    hardening.NoKinematicHardeningModel(),
                ),
            ),
            use_adjoint=False,
        )

        self.extract_grad = lambda m: [
            m.model.E.pvalue.grad.numpy(),
            m.model.flowrule.n.pvalue.grad.numpy(),
            m.model.flowrule.eta.pvalue.grad.numpy(),
            m.model.flowrule.s0.pvalue.grad.numpy(),
            m.model.flowrule.isotropic.R.pvalue.grad.numpy(),
            m.model.flowrule.isotropic.d.pvalue.grad.numpy(),
        ]

        self.times = torch.transpose(
            torch.tensor(
                np.array([np.linspace(0, 1, self.ntime) for i in range(self.nbatch)])
            ),
            1,
            0,
        )
        self.strains = torch.transpose(
            torch.tensor(
                np.array(
                    [np.linspace(0, 0.003, self.ntime) for i in range(self.nbatch)]
                )
            ),
            1,
            0,
        )
        self.stresses = torch.transpose(
            torch.tensor(
                np.array(
                    [np.linspace(0, 200.0, self.ntime) for i in range(self.nbatch)]
                )
            ),
            1,
            0,
        )
        self.temperatures = torch.zeros_like(self.strains)


class TestHardeningViscoplasticity(unittest.TestCase, CommonGradient):
    def setUp(self):
        self.ntime = 10
        self.nbatch = 10

        self.E = torch.tensor(100000.0)
        self.n = torch.tensor(5.2)
        self.eta = torch.tensor(110.0)
        self.R = torch.tensor(100.0)
        self.d = torch.tensor(5.1)
        self.C = torch.tensor(1000.0)
        self.g = torch.tensor(10.0)
        self.s0 = torch.tensor(10.0)

        self.p = [self.E, self.n, self.eta, self.s0, self.R, self.d, self.C, self.g]

        self.model_fn = lambda p: models.ModelIntegrator(
            models.InelasticModel(
                CP(p[0]),
                flowrules.IsoKinViscoplasticity(
                    CP(p[1]),
                    CP(p[2]),
                    CP(p[3]),
                    hardening.VoceIsotropicHardeningModel(CP(p[4]), CP(p[5])),
                    hardening.FAKinematicHardeningModel(CP(p[6]), CP(p[7])),
                ),
            ),
            use_adjoint=False,
        )

        self.extract_grad = lambda m: [
            m.model.E.pvalue.grad.numpy(),
            m.model.flowrule.n.pvalue.grad.numpy(),
            m.model.flowrule.eta.pvalue.grad.numpy(),
            m.model.flowrule.s0.pvalue.grad.numpy(),
            m.model.flowrule.isotropic.R.pvalue.grad.numpy(),
            m.model.flowrule.isotropic.d.pvalue.grad.numpy(),
            m.model.flowrule.kinematic.C.pvalue.grad.numpy(),
            m.model.flowrule.kinematic.g.pvalue.grad.numpy(),
        ]

        self.times = torch.transpose(
            torch.tensor(
                np.array([np.linspace(0, 1, self.ntime) for i in range(self.nbatch)])
            ),
            1,
            0,
        )
        self.strains = torch.transpose(
            torch.tensor(
                np.array(
                    [np.linspace(0, 0.003, self.ntime) for i in range(self.nbatch)]
                )
            ),
            1,
            0,
        )
        self.stresses = torch.transpose(
            torch.tensor(
                np.array(
                    [np.linspace(0, 200.0, self.ntime) for i in range(self.nbatch)]
                )
            ),
            1,
            0,
        )
        self.temperatures = torch.zeros_like(self.strains)


class TestHardeningViscoplasticityDamage(unittest.TestCase, CommonGradient):
    def setUp(self):
        self.ntime = 10
        self.nbatch = 10

        self.E = torch.tensor(100000.0)
        self.n = torch.tensor(5.2)
        self.eta = torch.tensor(110.0)
        self.R = torch.tensor(100.0)
        self.d = torch.tensor(5.1)
        self.C = torch.tensor(1000.0)
        self.g = torch.tensor(10.0)
        self.s0 = torch.tensor(10.0)
        self.A = torch.tensor(2000.0)
        self.xi = torch.tensor(6.5)
        self.phi = torch.tensor(1.7)

        self.p = [
            self.E,
            self.n,
            self.eta,
            self.s0,
            self.R,
            self.d,
            self.C,
            self.g,
            self.A,
            self.xi,
            self.phi,
        ]

        self.model_fn = lambda p: models.ModelIntegrator(
            models.InelasticModel(
                CP(p[0]),
                flowrules.IsoKinViscoplasticity(
                    CP(p[1]),
                    CP(p[2]),
                    CP(p[3]),
                    hardening.VoceIsotropicHardeningModel(CP(p[4]), CP(p[5])),
                    hardening.FAKinematicHardeningModel(CP(p[6]), CP(p[7])),
                ),
                dmodel=damage.HayhurstLeckie(CP(p[8]), CP(p[9]), CP(p[10])),
            ),
            use_adjoint=False,
        )

        self.extract_grad = lambda m: [
            m.model.E.pvalue.grad.numpy(),
            m.model.flowrule.n.pvalue.grad.numpy(),
            m.model.flowrule.eta.pvalue.grad.numpy(),
            m.model.flowrule.s0.pvalue.grad.numpy(),
            m.model.flowrule.isotropic.R.pvalue.grad.numpy(),
            m.model.flowrule.isotropic.d.pvalue.grad.numpy(),
            m.model.flowrule.kinematic.C.pvalue.grad.numpy(),
            m.model.flowrule.kinematic.g.pvalue.grad.numpy(),
            m.model.dmodel.A.pvalue.grad.numpy(),
            m.model.dmodel.xi.pvalue.grad.numpy(),
            m.model.dmodel.phi.pvalue.grad.numpy(),
        ]

        self.times = torch.transpose(
            torch.tensor(
                np.array([np.linspace(0, 1, self.ntime) for i in range(self.nbatch)])
            ),
            1,
            0,
        )
        self.strains = torch.transpose(
            torch.tensor(
                np.array([np.linspace(0, 0.03, self.ntime) for i in range(self.nbatch)])
            ),
            1,
            0,
        )
        self.stresses = torch.transpose(
            torch.tensor(
                np.array([np.linspace(0, 200, self.ntime) for i in range(self.nbatch)])
            ),
            1,
            0,
        )
        self.temperatures = torch.zeros_like(self.strains)


class TestChabocheViscoplasticity(unittest.TestCase, CommonGradient):
    def setUp(self):
        self.ntime = 10
        self.nbatch = 4

        self.E = torch.tensor(100000.0)
        self.n = torch.tensor(5.2)
        self.eta = torch.tensor(110.0)
        self.R = torch.tensor(100.0)
        self.d = torch.tensor(5.1)
        self.C = torch.tensor([1000.0, 750.0, 100.0])
        self.g = torch.tensor([10.0, 1.2, 8.6])
        self.s0 = torch.tensor(10.0)

        self.p = [self.E, self.n, self.eta, self.s0, self.R, self.d, self.C, self.g]

        self.model_fn = lambda p: models.ModelIntegrator(
            models.InelasticModel(
                CP(p[0]),
                flowrules.IsoKinViscoplasticity(
                    CP(p[1]),
                    CP(p[2]),
                    CP(p[3]),
                    hardening.VoceIsotropicHardeningModel(CP(p[4]), CP(p[5])),
                    hardening.ChabocheHardeningModel(CP(p[6]), CP(p[7])),
                ),
            ),
            use_adjoint=False,
        )

        self.extract_grad = lambda m: [
            m.model.E.pvalue.grad.numpy(),
            m.model.flowrule.n.pvalue.grad.numpy(),
            m.model.flowrule.eta.pvalue.grad.numpy(),
            m.model.flowrule.s0.pvalue.grad.numpy(),
            m.model.flowrule.isotropic.R.pvalue.grad.numpy(),
            m.model.flowrule.isotropic.d.pvalue.grad.numpy(),
            m.model.flowrule.kinematic.C.pvalue.grad.numpy(),
            m.model.flowrule.kinematic.g.pvalue.grad.numpy(),
        ]

        self.times = torch.transpose(
            torch.tensor(
                np.array([np.linspace(0, 1, self.ntime) for i in range(self.nbatch)])
            ),
            1,
            0,
        )
        self.strains = torch.transpose(
            torch.tensor(
                np.array(
                    [np.linspace(0, 0.003, self.ntime) for i in range(self.nbatch)]
                )
            ),
            1,
            0,
        )
        self.stresses = torch.transpose(
            torch.tensor(
                np.array(
                    [np.linspace(0, 200.0, self.ntime) for i in range(self.nbatch)]
                )
            ),
            1,
            0,
        )
        self.temperatures = torch.zeros_like(self.strains)
