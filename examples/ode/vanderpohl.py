#!/usr/bin/env python3

"""
  This example demonstrate the ability of the backward Euler integration
  option to accurately integrate a very stiff system of ODEs.  The example
  system used here is the Van der Pol system from Manichev et al. 2019.

  .. math::

    \\dot{y}_0 = y_1
    \\dot{y}_1 = -y_0 + \\mu \\left(1 - y_0^2 \\right) y_1

  with initial conditions

  .. math::

    y_0(0) = -1
    y_1(0) = 1

  and :math:`t \\in [0,4.2\\mu].

  This system of ODEs is parameterized by :math:`\\mu`.  For
  :math:`mu = 1.0` the system is not stiff and can be integrated by
  either forward or backward methods.  For :math:`\\mu=14` the equations
  begin to become stiff and the forward Euler method becomes inaccurate.
  For :math:`\\mu=30` the equations are very stiff and can only
  be integrated with the backward Euler method.
"""

import sys

sys.path.append("../..")

import numpy as np
import torch

import matplotlib.pyplot as plt

from pyoptmat import ode

torch.set_default_tensor_type(torch.DoubleTensor)


class VanderPolODE(torch.nn.Module):
    """
    From Manichev et al 2019

      x0 = [-1, 1]
      t = [0, 4.2*mu)
    """

    def __init__(self, mu):
        super().__init__()
        self.mu = torch.tensor(mu)

    def forward(self, t, y):
        f = torch.empty(y.shape)
        f[..., 0] = y[..., 1]
        f[..., 1] = -y[..., 0] + self.mu * (1.0 - y[..., 0] ** 2.0) * y[..., 1]

        df = torch.empty(y.shape + y.shape[-1:])
        df[..., 0, 0] = 0
        df[..., 0, 1] = 1
        df[..., 1, 0] = -1 - 2.0 * self.mu * y[..., 0] * y[..., 1]
        df[..., 1, 1] = self.mu * (1.0 - y[..., 0] ** 2.0)

        return f, df


if __name__ == "__main__":
    # Common inputs
    y0 = torch.tensor([[-1.0, 1]])
    period = lambda m: (3.0 - 2 * np.log(2)) * m + 2.0 * np.pi / mu ** (1.0 / 3)

    # Number of vectorized time steps
    time_chunk = 10

    # Test for non-stiff version
    mu = 1.0
    times = torch.linspace(0, period(mu) * 10, 10000).unsqueeze(-1)

    model = VanderPolODE(mu)

    res_exp = ode.odeint(model, y0, times, method="forward-euler",
            block_size = time_chunk, guess_type = "previous")
    res_imp = ode.odeint(
        model, y0, times, method="backward-euler",
        block_size = time_chunk, guess_type = "previous"
    )

    plt.figure()
    plt.plot(times, res_exp[:, 0, 0], label="Forward")
    plt.plot(times, res_imp[:, 0, 0], label="Backward")
    plt.xlabel("time")
    plt.ylabel(r"$y_0$")
    plt.legend(loc="best")
    plt.title("Not stiff")
    plt.show()

    # Test for moderately stiff version
    mu = 14.0
    times = torch.linspace(0, period(mu) * 10, 10000).unsqueeze(-1)

    model = VanderPolODE(mu)

    res_exp = ode.odeint(model, y0, times, method="forward-euler",
            block_size = time_chunk, guess_type = "previous")
    res_imp = ode.odeint(
        model, y0, times, method="backward-euler",
        block_size = time_chunk, guess_type = "previous"
    )

    plt.figure()
    plt.plot(times, res_exp[:, 0, 0], label="Forward")
    plt.plot(times, res_imp[:, 0, 0], label="Backward")
    plt.legend(loc="best")
    plt.xlabel("time")
    plt.ylabel(r"$y_0$")
    plt.title("Moderately stiff")
    plt.show()

    # Test for highly stiff version (explicit just explodes)
    mu = 30.0
    times = torch.linspace(0, period(mu) * 10 * 10.0 / 16, 20000).unsqueeze(-1)

    model = VanderPolODE(mu)

    res_imp = ode.odeint(
        model, y0, times, method="backward-euler",
        block_size = time_chunk, guess_type = "previous"
    )

    plt.figure()
    plt.plot(times, res_imp[:, 0, 0], label="Backward")
    plt.legend(loc="best")
    plt.xlabel("time")
    plt.ylabel(r"$y_0$")
    plt.title("Very stiff")
    plt.show()
