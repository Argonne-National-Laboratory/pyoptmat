import unittest

import torch
import torch.nn
from torch.nn import Parameter

from pyoptmat import ode, utility

torch.set_default_tensor_type(torch.DoubleTensor)


class SimpleODE(torch.nn.Module):
    def __init__(self, A, B):
        super().__init__()

        self.A = Parameter(A)
        self.B = Parameter(B)

    def forward(self, t, y):
        return self.A * y + self.B, self.A * torch.ones_like(y).unsqueeze(-1)


class IntegratedODE(torch.nn.Module):
    def __init__(self, model, y0, y0_param=False, adjoint=False):
        super().__init__()

        self.model = model

        if y0_param:
            self.y0 = Parameter(y0)
        else:
            self.y0 = y0

        if adjoint:
            self.method = ode.odeint_adjoint
        else:
            self.method = ode.odeint

    def forward(self, times):
        return self.method(self.model, 2.0 * self.y0, times)


class TestSimple(unittest.TestCase):
    def setUp(self):
        self.A = torch.tensor(2.0)
        self.B = torch.tensor(1.5)

        self.nbatch = 10

        self.y0 = torch.rand((10, 1))

        self.tmax = 1.0
        self.ntime = 20
        self.times = (
            torch.linspace(0, self.tmax, self.ntime)
            .unsqueeze(-1)
            .expand((self.ntime, self.nbatch))
        )

        self.model = SimpleODE(self.A, self.B)

    def test_adjoint(self):
        model_ad = IntegratedODE(self.model, self.y0, y0_param=True, adjoint=False)
        res_ad = torch.norm(model_ad(self.times))
        res_ad.backward()
        ad_res = model_ad.y0.grad.detach().clone()

        self.model.zero_grad()

        model_adjoint = IntegratedODE(self.model, self.y0, y0_param=True, adjoint=True)
        res_adjoint = torch.norm(model_adjoint(self.times))
        res_adjoint.backward()
        adjoint_res = model_adjoint.y0.grad.detach().clone()

        self.assertTrue(torch.allclose(ad_res, adjoint_res))
