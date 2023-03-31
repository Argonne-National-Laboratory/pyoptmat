import unittest

import numpy as np

import torch
import torch.nn

from pyoptmat import ode

torch.set_default_tensor_type(torch.DoubleTensor)


class LogisticODE(torch.nn.Module):
    def __init__(self, r, K):
        super().__init__()
        self.r = torch.tensor(r)
        self.K = torch.tensor(K)

    def forward(self, t, y):
        return (
            self.r * (1.0 - y / self.K) * y,
            (self.r - (2 * self.r * y) / self.K)[..., None],
        )

    def exact(self, t, y0):
        return (
            self.K
            * torch.exp(self.r * t)
            * y0[..., 0]
            / (self.K + (torch.exp(self.r * t) - 1) * y0[..., 0])
        )[..., None]


class TestBatchSimple(unittest.TestCase):
    def setUp(self):
        self.r = 1.0
        self.K = 1.0

        self.model = LogisticODE(self.r, self.K)
        self.nbatch = 20
        self.y0 = torch.linspace(0.51, 0.76, self.nbatch).reshape(self.nbatch, 1)

        self.nsteps = 100

        self.times = torch.tensor(
            np.array([np.linspace(0, 7.5, self.nsteps) for i in range(self.nbatch)]).T
        )

    def test_forward_default(self):
        exact = self.model.exact(self.times, self.y0)
        numerical = ode.odeint(self.model, self.y0, self.times, method="forward-euler")

        self.assertTrue(torch.max(torch.abs((numerical - exact) / exact)) < 1.0e-2)

    def test_backward_default(self):
        exact = self.model.exact(self.times, self.y0)
        numerical = ode.odeint(self.model, self.y0, self.times, method="backward-euler")

        self.assertTrue(torch.max(torch.abs((numerical - exact) / exact)) < 1.0e-2)


class TestBatchSimpleChunkTime(unittest.TestCase):
    def setUp(self):
        self.r = 1.0
        self.K = 1.0

        self.model = LogisticODE(self.r, self.K)
        self.nbatch = 20
        self.y0 = torch.linspace(0.51, 0.76, self.nbatch).reshape(self.nbatch, 1)

        self.nsteps = 100

        self.tchunk = 3

        self.times = torch.tensor(
            np.array([np.linspace(0, 7.5, self.nsteps) for i in range(self.nbatch)]).T
        )

    def test_forward_default(self):
        exact = self.model.exact(self.times, self.y0)
        numerical = ode.odeint(
            self.model,
            self.y0,
            self.times,
            method="forward-euler",
            block_size=self.tchunk,
        )

        self.assertTrue(torch.max(torch.abs((numerical - exact) / exact)) < 1.0e-2)

    def test_backward_default(self):
        exact = self.model.exact(self.times, self.y0)
        numerical = ode.odeint(
            self.model,
            self.y0,
            self.times,
            method="backward-euler",
            block_size=self.tchunk,
        )

        self.assertTrue(torch.max(torch.abs((numerical - exact) / exact)) < 1.0e-2)


class FallingODE(torch.nn.Module):
    """
    From ORNL/TM-9912

    Parameters are:
      m = 0.25
      w = 8.0
      k = 2.0

      y0 = 0
    """

    def __init__(self, m, w, k):
        super().__init__()
        self.m = torch.tensor(m)
        self.w = torch.tensor(w)
        self.k = torch.tensor(k)

    def forward(self, t, y):
        f = torch.empty(y.shape)
        f[..., 0] = y[..., 1]
        f[..., 1] = (self.w - self.k * y[..., 1]) / self.m

        df = torch.zeros(y.shape + y.shape[-1:])
        df[..., 0, 1] = 1
        df[..., 1, 1] = -self.k / self.m

        return f, df

    def exact(self, t, y0):
        return torch.stack(
            (
                4.0 * (t + torch.exp(-8.0 * t) / 8.0 - 1.0 / 8),
                4.0 * (1 - torch.exp(-8.0 * t)),
            ),
            dim=1,
        ).permute(0, 2, 1)


class TestFallingBatch(unittest.TestCase):
    def setUp(self):
        self.m = 1.0 / 4.0
        self.w = 8.0
        self.k = 2.0

        self.nbatch = 20
        self.y0 = torch.zeros(self.nbatch, 2)

        self.nsteps = 1000
        self.nchunk = 10
        self.times = torch.tensor(
            np.array([np.linspace(0, 5.0, self.nsteps) for i in range(self.nbatch)]).T
        )

        self.model = FallingODE(self.m, self.w, self.k)

    def test_forward(self):
        exact = self.model.exact(self.times, self.y0)
        numerical = ode.odeint(
            self.model,
            self.y0,
            self.times,
            method="forward-euler",
            block_size=self.nchunk,
        )

        self.assertTrue(torch.max(torch.abs((numerical[1:] - exact[1:]))) < 1.0e-1)

    def test_backward(self):
        exact = self.model.exact(self.times, self.y0)
        numerical = ode.odeint(
            self.model,
            self.y0,
            self.times,
            method="backward-euler",
            block_size=self.nchunk,
        )

        self.assertTrue(torch.max(torch.abs((numerical[1:] - exact[1:]))) < 1.0e-1)


class FallingParameterizedODE(torch.nn.Module):
    """
    From ORNL/TM-9912

    Parameters are:
      m = 0.25
      w = 8.0
      k = 2.0

      y0 = 0
    """

    def __init__(self, m, w, k):
        super().__init__()
        self.m = torch.nn.Parameter(torch.tensor(m))
        self.w = torch.nn.Parameter(torch.tensor(w))
        self.k = torch.nn.Parameter(torch.tensor(k))

    def forward(self, t, y):
        f = torch.empty(y.shape)
        f[..., 0] = y[..., 1].clone()
        f[..., 1] = (self.w - self.k * y[..., 1].clone()) / self.m

        df = torch.zeros(y.shape + y.shape[-1:])
        df[..., 0, 1] = 1
        df[..., 1, 1] = -self.k / self.m

        return f, df

    def exact(self, t, y0):
        return torch.stack(
            (
                4.0 * (t + torch.exp(-8.0 * t) / 8.0 - 1.0 / 8),
                4.0 * (1 - torch.exp(-8.0 * t)),
            ),
            dim=1,
        )


def run_model(model, nbatch=4, method="forward-euler", nsteps=10):
    y0 = torch.zeros(nbatch, 2)

    times = torch.tensor(
        np.array([np.linspace(0, 1.0, nsteps) for i in range(nbatch)]).T
    )

    numerical = ode.odeint_adjoint(model, y0, times, method=method)

    return torch.norm(numerical)


class TestGradient(unittest.TestCase):
    def setUp(self):
        self.m = 1.0 / 4.0
        self.w = 8.0
        self.k = 2.0

    def test_grad_forward(self):
        model = FallingParameterizedODE(self.m, self.w, self.k)
        res1 = run_model(model, method="forward-euler")
        res1.backward()

        r0 = res1.data.numpy()

        dm = model.m.grad.numpy()
        dw = model.w.grad.numpy()
        dk = model.k.grad.numpy()

        with torch.no_grad():
            eps = 1.0e-6
            dmp = (
                (
                    run_model(
                        FallingParameterizedODE(self.m + self.m * eps, self.w, self.k),
                        method="forward-euler",
                    )
                    - r0
                )
                / (eps * self.m)
            ).numpy()
            dwp = (
                (
                    run_model(
                        FallingParameterizedODE(self.m, self.w * (1 + eps), self.k),
                        method="forward-euler",
                    )
                    - r0
                )
                / (eps * self.w)
            ).numpy()
            dwk = (
                (
                    run_model(
                        FallingParameterizedODE(self.m, self.w, self.k * (1 + eps)),
                        method="forward-euler",
                    )
                    - r0
                )
                / (eps * self.k)
            ).numpy()

        self.assertAlmostEqual(dm, dmp, places=3)
        self.assertAlmostEqual(dw, dwp, places=3)
        self.assertAlmostEqual(dk, dwk, places=3)

    def test_grad_backward(self):
        model = FallingParameterizedODE(self.m, self.w, self.k)

        res1 = run_model(model, method="backward-euler")
        res1.backward()

        r0 = res1.data.numpy()

        dm = model.m.grad.numpy()
        dw = model.w.grad.numpy()
        dk = model.k.grad.numpy()

        with torch.no_grad():
            eps = 1.0e-6
            dmp = (
                (
                    run_model(
                        FallingParameterizedODE(self.m + self.m * eps, self.w, self.k),
                        method="backward-euler",
                    )
                    - r0
                )
                / (eps * self.m)
            ).numpy()
            dwp = (
                (
                    run_model(
                        FallingParameterizedODE(self.m, self.w * (1 + eps), self.k),
                        method="backward-euler",
                    )
                    - r0
                )
                / (eps * self.w)
            ).numpy()
            dwk = (
                (
                    run_model(
                        FallingParameterizedODE(self.m, self.w, self.k * (1 + eps)),
                        method="backward-euler",
                    )
                    - r0
                )
                / (eps * self.k)
            ).numpy()

        self.assertAlmostEqual(dm, dmp, places=3)
        self.assertAlmostEqual(dw, dwp, places=3)
        self.assertAlmostEqual(dk, dwk, places=3)
