"""
    Improved scaling functions with a better user interface

    These mimic the simple scaling functions in :py:mod:`pyoptmat.optimize`
    but provide a nicer interface for going from "actual" parameters to
    the scaled parameters and vice-versa.  However they should be
    one-to-one replacements for the original, simpler functions.
"""

import torch
from torch import nn


class ScalingFunction(nn.Module):
    """
    Common superclass for scaling functions, defines required
    methods.
    """

    def forward(self, x):
        """
        Passes through a function to the `scale` method

        Args:
            x (torch.tensor):   scaled parameter values
        """
        return self.scale(x)

    def scale(self, x):
        """
        Converts the scaled parameter values to the actual
        parameter scale

        Args:
            x (torch.tensor):   scaled parameter values
        """
        raise NotImplementedError("Method pure virtual")

    def unscale(self, x):
        """
        Converts the unscaled parameter values into the
        scaled values, for initialization

        Args:
            x (torch.tensor):   unscaled parameter values
        """
        raise NotImplementedError("Method pure virtual")

    def scale_stat(self, loc, scale):
        """
        Provide information on the actual parameter distribution,
        assuming the scaled properties are normally distributed
        and parameterized by loc and scale.

        Args:
            loc (torch.tensor):     location of the scaled parameters
            scale (torch.tensor):   scale of the scaled parameters
        """
        raise NotImplementedError("Method pure virtual")


class SimpleScalingFunction(ScalingFunction):
    """
    Scaling function where the unscaled parameters are

    .. math::

        y = s x

    where :math:`s` is the scale factor

    Args:
        s (torch.tensor):   scale factor
    """

    def __init__(self, s):
        super().__init__()
        self.s = s

    def scale(self, x):
        """
        Converts the scaled parameter values to the actual
        parameter scale

        Args:
            x (torch.tensor):   scaled parameter values
        """
        return self.s * x

    def unscale(self, x):
        """
        Converts the unscaled parameter values into the
        scaled values, for initialization

        Args:
            x (torch.tensor):   unscaled parameter values
        """
        return x / self.s

    def scale_stat(self, loc, scale):
        """
        Print some information about how the normally-distributed
        scaled parameters transform into "real" parameter space

        Args:
            loc (torch.tensor):     scaled loc
            scale (torch.tensor):   scaled scale
        """
        mean = self.scale(loc)
        std = scale * self.s

        msg = ""
        msg += "Normal distribution with:\n"
        msg += "\tloc:\n"
        msg += "\t" + str(mean) + "\n"
        msg += "\tscale:\n"
        msg += "\t" + str(std)

        return msg


class BoundedScalingFunction(ScalingFunction):
    """
    Scaling function where the unscaled parameters are

    .. math::

        y = x \\left(u - l \\right) + l

    where :math:`l` is the lower bound on the parameters
    and :math:`u` is the upper bound.

    Args:
        l (torch.tensor):   lower bound on the parameters
        u (torch.tensor):   upper bound on the parameter
        clamp (optional):   if True also clip the values to remain in range
    """

    def __init__(self, l, u, clamp=True):
        super().__init__()
        self.l = l
        self.u = u
        self.clamp = clamp

    def scale(self, x):
        """
        Converts the scaled parameter values to the actual
        parameter scale

        Args:
            x (torch.tensor):   scaled parameter values
        """
        if self.clamp:
            xp = torch.clamp(x, 0.0, 1.0)
        else:
            xp = x

        return xp * (self.u - self.l) + self.l

    def unscale(self, x):
        """
        Converts the unscaled parameter values into the
        scaled values, for initialization

        Args:
            x (torch.tensor):   unscaled parameter values
        """
        return (x - self.l) / (self.u - self.l)

    def scale_stat(self, loc, scale):
        """
        Print some information about how the normally-distributed
        scaled parameters transform into "real" parameter space

        Args:
            loc (torch.tensor):     scaled loc
            scale (torch.tensor):   scaled scale
        """
        mean = self.scale(loc)
        std = scale * (self.u - self.l)

        msg = ""
        msg += "Normal distribution with:\n"
        msg += "\tloc:\n"
        msg += "\t" + str(mean) + "\n"
        msg += "\tscale:\n"
        msg += "\t" + str(std)

        return msg


class LogBoundedScalingFunction(ScalingFunction):
    """
    Scaling function where the unscaled parameters are

    .. math::

        y = l**(1-x)*u**x

    where :math:`l` is the lower bound on the parameters
    and :math:`u` is the upper bound.

    That is, the natural parameter is log transformed and then
    scaled between two (log transformed) bounds

    The resulting distribution is log normal

    Args:
        l (torch.tensor):   lower bound on the parameters
        u (torch.tensor):   upper bound on the parameter
        clamp (optional):   if True also clip the values to remain in range
    """

    def __init__(self, l, u, clamp=True):
        super().__init__()
        self.l = l
        self.u = u
        self.clamp = clamp

    def scale(self, x):
        """
        Converts the scaled parameter values to the actual
        parameter scale

        Args:
            x (torch.tensor):   scaled parameter values
        """
        if self.clamp:
            xp = torch.clamp(x, 0.0, 1.0)
        else:
            xp = x

        return self.l ** (1 - xp) * self.u**xp

    def unscale(self, x):
        """
        Converts the unscaled parameter values into the
        scaled values, for initialization

        Args:
            x (torch.tensor):   unscaled parameter values
        """
        return (torch.log(x) - torch.log(self.l)) / (
            torch.log(self.u) - torch.log(self.l)
        )

    def scale_stat(self, loc, scale):
        """
        Print some information about how the normally-distributed
        scaled parameters transform into "real" parameter space

        Args:
            loc (torch.tensor):     scaled loc
            scale (torch.tensor):   scaled scale
        """
        A = torch.log(self.u) - torch.log(self.l)
        B = torch.log(self.l)

        mean = A * loc + B
        sp = A * scale

        amean = torch.exp(mean + sp**2 / 2.0)

        msg = ""
        msg += "Log normal distribution with:\n"
        msg += "\tloc:\n"
        msg += "\t" + str(mean) + "\n"
        msg += "\tscale:\n"
        msg += "\t" + str(sp) + "\n"
        msg += "\tand mean values:\n"
        msg += "\t" + str(amean)

        return msg
