"""
  Various utility functions used in the rest of the modules.  This includes
  basic mathematical functions, routines used in the tests, and various
  visualization routines.
"""

from functools import reduce, wraps

import numpy as np

import matplotlib.pyplot as plt

import torch
from torch import nn
import functorch


def compose(f1, f2):
    """A function composition operator...

    Args:
        f1 (function): first function
        f2 (function): second function

    Returns:
        function: f2(f1)
    """
    return f2(f1)


def jacobianize(argnums=None):
    """Decorator that adds the multibatched Jacobian to a function

    Assumes that the function itself has only a single dimension, the last in the shape

    By default provides Jacobian for all *args, but argnums can be set to limit this
    to certain arguments
    """

    def inner_jacobianize(fn):
        @wraps(fn)
        def wrapper(*args, argnums=argnums, **kwargs):
            if argnums is None:
                argnums = tuple(range(len(args)))
            res = fn(*args, **kwargs)
            repeats = res.dim() - 1
            D = reduce(
                compose,
                [fn, lambda x: functorch.jacrev(x, argnums=argnums)]
                + [functorch.vmap] * repeats,
            )

            return res, D(*args, **kwargs)

        return wrapper

    return inner_jacobianize


def mbmm(A1, A2):
    """
    Batched matrix-matrix multiplication with several batch dimensions
    """
    return torch.einsum("...ik,...kj->...ij", A1, A2)


def visualize_variance(strain, stress_true, stress_calc, alpha=0.05):
    """
    Visualize variance for batched examples

    Args:
      strain (torch.tensor):        input strain
      stress_true (torch.tensor):   actual stress values
      stress_calc (torch.tensor):   simulated stress values

    Keyword Args:
      alpha (float): alpha value for shading
    """
    ntrue = stress_true.shape[1]
    max_true, _ = stress_true.kthvalue(int(ntrue * (1 - alpha)), dim=1)
    min_true, _ = stress_true.kthvalue(int(ntrue * alpha), dim=1)
    mean_true = torch.mean(stress_true, 1)

    npred = stress_calc.shape[1]
    max_pred, _ = stress_calc.kthvalue(int(npred * (1 - alpha)), dim=1)
    min_pred, _ = stress_calc.kthvalue(int(npred * alpha), dim=1)
    mean_pred = torch.mean(stress_calc, 1)

    plt.figure()
    plt.plot(strain.numpy(), mean_true.numpy(), "k-", label="Actual mean")
    plt.fill_between(
        strain.numpy(),
        min_true.numpy(),
        max_true.numpy(),
        alpha=0.5,
        label="Actual range",
    )
    plt.plot(strain.numpy(), mean_pred.numpy(), "k--", label="Predicted mean")
    plt.fill_between(
        strain.numpy(),
        min_pred.numpy(),
        max_pred.numpy(),
        alpha=0.5,
        label="Predicted range",
    )
    plt.xlabel("Strain (mm/mm)")
    plt.ylabel("Stress (MPa)")
    plt.legend(loc="best")

    plt.show()


def batch_differentiate(fn, x0, eps=1.0e-6, nbatch_dim=1):
    """
    New numerical differentiation function to handle the batched-model cases

    This version handles arbitrary batch sizes

    Args:
      fn (torch.tensor):    function to differentiate via finite differences
      x0 (torch.tensor):    point at which to take the numerical derivative

    Keyword Args:
      eps (float):          perturbation to use

    Returns:
      torch.tensor:         finite difference approximation to
                            :math:`\\frac{df}{dx}|_{x_0}`
    """
    v0 = fn(x0)
    bs = v0.shape[:nbatch_dim]

    s1 = v0.shape[nbatch_dim:]
    s2 = x0.shape[nbatch_dim:]
    squeeze1 = False
    squeeze2 = False

    if len(s1) == 0:
        s1 = (1,)
        squeeze1 = True

    if len(s2) == 0:
        s2 = (1,)
        squeeze2 = True

    d = torch.empty(bs + s1 + s2, device=x0.device)

    x0 = x0.reshape(bs + s2)
    v0 = v0.reshape(bs + s1)

    for i2 in np.ndindex(s2):
        dx = torch.zeros_like(x0)
        inc = torch.abs(x0[..., i2]) * eps
        inc[inc < eps] = eps
        dx[..., i2] = inc

        x = x0 + dx
        if squeeze2:
            x = x.squeeze(-1)

        v1 = fn(x).reshape(bs + s1)
        d[..., i2] = (v1 - v0).unsqueeze(-1).expand(d[..., i2].shape) / inc.unsqueeze(
            -2
        ).expand(d[..., i2].shape)

    if squeeze1 and squeeze2:
        d = d.squeeze(-1).squeeze(-1)
    elif squeeze2:
        d = d.squeeze(-1)
    elif squeeze1:
        d = d.squeeze(-len(s2) - 1)

    return d


def new_differentiate(fn, x0, eps=1.0e-6):
    """
    New numerical differentiation function to handle the batched-model cases

    Args:
      fn (torch.tensor):    function to differentiate via finite differences
      x0 (torch.tensor):    point at which to take the numerical derivative

    Keyword Args:
      eps (float):          perturbation to use

    Returns:
      torch.tensor:         finite difference approximation to
                            :math:`\\frac{df}{dx}|_{x_0}`
    """
    v0 = fn(x0)
    nbatch = v0.shape[0]

    s1 = v0.shape[1:]
    if len(s1) == 0:
        s1 = (1,)
    fs1 = (nbatch,) + s1

    s2 = x0.shape[1:]
    if len(s2) == 0:
        flatten = True
        s2 = (1,)
    else:
        flatten = False
    fs2 = (nbatch,) + s2

    d = torch.empty((nbatch,) + s1 + s2, device=x0.device)

    v0 = v0.reshape(fs1)

    for i2 in np.ndindex(s2):
        dx = torch.zeros_like(x0.reshape(fs2))
        inc = torch.abs(x0[..., i2]) * eps
        inc[inc < eps] = eps
        dx[..., i2] = inc

        if flatten:
            v1 = fn(x0 + dx[:, 0]).reshape(fs1)
        else:
            v1 = fn(x0 + dx).reshape(fs1)

        d[:, :, i2] = torch.unsqueeze(((v1 - v0) / inc), 2)

    return d


def differentiate(fn, x0, eps=1.0e-6):
    """
    Numerical differentiation used in the tests, old version does not
    handle batched input

    Args:
      fn (torch.tensor):    function to differentiate via finite differences
      x0 (torch.tensor):    point at which to take the numerical derivative

    Keyword Args:
      eps (float):          perturbation to use

    Returns:
      torch.tensor:         finite difference approximation to
                            :math:`\\frac{df}{dx}|_{x_0}`
    """
    v0 = fn(x0)
    nbatch = v0.shape[0]
    s1 = v0.shape[1:]
    s2 = x0.shape[1:]

    if len(s1) == 0 and len(s2) == 0:
        d = torch.empty((nbatch,))
    elif len(s1) == 0:
        d = torch.empty((nbatch,) + s2)
    elif len(s2) == 0:
        d = torch.empty((nbatch,) + s1)
    else:
        d = torch.empty((nbatch,) + s1 + s2)

    for i2 in np.ndindex(s2):
        dx = torch.zeros_like(x0)
        if len(s2) == 0:
            dx = torch.abs(x0) * eps
        else:
            inc = torch.abs(x0[..., i2]) * eps
            inc[inc < eps] = eps
            dx[..., i2] = inc
        v1 = fn(x0 + dx)

        if len(s1) == 0 and len(s2) == 0:
            d = (v1 - v0) / dx
        elif len(s2) == 0:
            d[...] = (v1 - v0) / dx.reshape(v1.shape)
        elif len(s1) == 0:
            d[..., i2] = ((v1 - v0) / dx.reshape(v1.shape))[:, None]
        else:
            d[..., i2] = ((v1 - v0) / dx)[..., None]

    return d


class ArbitraryBatchTimeSeriesInterpolator(nn.Module):
    """
    Interpolate :code:`data` located at discrete :code:`times`
    linearly to point :code:`t`.

    This version handles batched of arbitrary size -- only the rightmost
    batch dimension must agree with the input data.  All other dimensions are
    broadcast.

    Args:
      times (torch.tensor):     input time series as a code:`(ntime,nbatch)`
                                array
      values (torch.tensor):    input values series as a code:`(ntime,nbatch)`
                                array
    """

    def __init__(self, times, data):
        super().__init__()
        # Transpose to put the shared dimension first
        self.times = times.t()
        self.values = data.t()

        self.slopes = torch.diff(self.values, dim=-1) / torch.diff(self.times, dim=-1)

    def forward(self, t):
        """
        Calculate the linearly-interpolated current values

        Args:
          t (torch.tensor):   batched times as :code:`(...,nbatch,)` array

        Returns:
          torch.tensor:       batched values at :code:`t`
        """
        squeeze = t.dim() == 1

        t = t.t()  # Transpose so the common dimension is first
        if squeeze:
            t = t.unsqueeze(-1)

        # Which dimensions to offset -- likely there is some efficiency
        # to be gained by flipping these, but probably depends on the
        # input shape
        d1 = -1
        d2 = -2

        offsets = t.unsqueeze(d1) - self.times[..., :-1].unsqueeze(d2)

        poss = self.slopes.unsqueeze(d2) * offsets + self.values[..., :-1].unsqueeze(d2)

        locs = torch.logical_and(
            t.unsqueeze(d1) <= self.times[..., 1:].unsqueeze(d2),
            t.unsqueeze(d1) > self.times[..., :-1].unsqueeze(d2),
        )

        # Now we need to fix up the stupid left side
        locs[..., 0] = torch.logical_or(
            locs[..., 0], t == self.times[..., 0].unsqueeze(-1)
        )

        if squeeze:
            return poss[locs].squeeze(-1).t()

        return poss[locs].reshape(t.shape).t()


class BatchTimeSeriesInterpolator(nn.Module):
    """
    Interpolate :code:`data` located at discrete :code:`times`
    linearly to point :code:`t`.

    This version handles batched input

    Precache a lot of the work required to interpolate in time vs
    :func:`pyoptmat.utility.timeseries_interpolate_batch_times`

    Args:
      times (torch.tensor):     input time series as a code:`(ntime,nbatch)`
                                array
      values (torch.tensor):    input values series as a code:`(ntime,nbatch)`
                                array
    """

    def __init__(self, times, data):
        super().__init__()
        self.times = times
        self.values = data

        self.slopes = torch.diff(self.values, dim=0) / torch.diff(self.times, dim=0)

    def forward(self, t):
        """
        Calculate the linearly-interpolated current values

        Args:
          t (torch.tensor):   batched times as :code:`(nbatch,)` array

        Returns:
          torch.tensor:       batched values at :code:`t`
        """
        gi = torch.remainder(
            torch.sum((self.times - t) <= 0, dim=0), self.times.shape[0]
        )
        return torch.diagonal(self.values[gi - 1]) + torch.diagonal(
            self.slopes[gi - 1]
        ) * (t - torch.diagonal(self.times[gi - 1]))


class CheaterBatchTimeSeriesInterpolator(nn.Module):
    """
    Interpolate :code:`data` located at discrete :code:`times`
    linearly to point :code:`t`.

    Precache a lot of the work required to interpolate in time vs
    :func:`pyoptmat.utility.timeseries_interpolate_batch_times`

    This is the cheater version specifically for our structured problems where
    if you figure out where one time point index is relative to the provided
    time points then you can use that index for all the other points
    in the batch.  This won't work in general, but works fine here.

    Args:
      times (torch.tensor):     input time series as a :code:`(ntime,nbatch)`
                                array
      values (torch.tensor):    input values series as a :code:`(ntime,nbatch)`
                                array
    """

    def __init__(self, times, data):
        super().__init__()
        self.times = times
        self.values = data

        self.slopes = torch.diff(self.values, dim=0) / torch.diff(self.times, dim=0)

    def forward(self, t):
        """
        Calculate the linearly-interpolated current values

        Args:
          t (torch.tensor):   batched times as :code:`(nbatch,)` array

        Returns:
          torch.tensor:       batched values at :code:`t`
        """
        gi = torch.maximum(
            torch.argmax((self.times[:, 0] >= t[0]).type(torch.uint8)),
            torch.tensor(1, device=t.device),
        )

        return self.values[gi - 1] + self.slopes[gi - 1] * (t - self.times[gi - 1])


def timeseries_interpolate_batch_times(times, values, t):
    """
    Interpolate the time series defined by X to the times defined by t

    This version handles batched input

    Args:
      times (torch.tensor):     input time series as a :code:`(ntime,nbatch)`
                                array
      values (torch.tensor):    input value series as a :code:`(ntime,nbatch)`
                                array
      t (torch.tensor):         batch times as a :code:`(nbatch,)` array

    Returns:
      torch.tensor:             Interpolated values as a :code:`(nbatch,)` array
    """
    gi = torch.remainder(torch.sum((times - t) <= 0, dim=0), times.shape[0])
    y2 = torch.diagonal(values[gi])
    y1 = torch.diagonal(values[gi - 1])
    t2 = torch.diagonal(times[gi])
    t1 = torch.diagonal(times[gi - 1])

    slopes = (y2 - y1) / (t2 - t1)
    return y1 + slopes * (t - t1)


def timeseries_interpolate_single_times(times, values, t):
    """
    Interpolate the time series defined by X to the times defined by t

    This version does *not* handle batched input

    Args:
      times (torch.tensor):     input time series as a :code:`(ntime,)` array
      values (torch.tensor):    input value series as a :code:`(ntime,nbatch)`
                                array
      t (torch.tensor):         times as a scalar

    Returns:
      torch.tensor:             interpolated values as a :code:`(nbatch,)` array
    """
    gi = torch.remainder(torch.sum((times - t) <= 0, dim=0), times.shape[0])
    slopes = (values[gi] - values[gi - 1]) / (times[gi, None] - times[gi - 1, None])
    return values[gi - 1] + slopes * (t - times[gi - 1])


def heaviside(X):
    """
    A pytorch-differentiable version of the Heaviside function

    .. math::

      H\\left(x\\right) = \\frac{\\operatorname{sign}(x) + 1)}{2}

    Args:
      X (torch.tensor): tensor input

    Returns:
      torch.tensor:     the Heaviside function of the input
    """
    return (torch.sign(X) + 1.0) / 2.0


def macaulay(X):
    """
    A pytorch-differentiable version of the Macualay bracket

    .. math::

      M\\left(x\\right) = x H\\left(x\\right)

    Args:
      X (torch.tensor): tensor input

    Returns:
      torch.tensor:     the Macaulay bracket applied to the input
    """
    return X * heaviside(X)


def add_id(df):
    """
    Add the identity to a tensor with the shape of the Jacobian

    Args:
      df (torch.tensor):  batched `(n,m,m)` tensor

    Returns:
      torch.tensor:       :code:`df` plus a batched identity of the right shape
    """
    return df + torch.eye(df.shape[1], device=df.device).reshape(
        (1,) + df.shape[1:]
    ).repeat(df.shape[0], 1, 1)
