"""
  The idea of the objects here, derived from |TP|
  is to provide temperature-dependent versions of raw model parameters.

  These scaling rules are functions of some "actual" parameters, either
  PyTorch or Pyro parameters, which are actually optimized.  These scaling
  rules somehow map a collection of these parameters to a temperature
  dependent property.

  On top of this temperature scaling, these classes can also, optionally
  be used to do numerical scaling of the underlying parameters to
  achieve better numerical stability.  So, in the end, mathematically these
  classes are functions of the kind

  .. math ::

    v = s(t(T;p))

  where :math:`s` is a scaling function with fixed parameters aimed at
  helping numerical scaling and :math:`t` is the actual temperature
  scaling function, parameterized with a set of parameters :math:`p` which
  will be varied during optimization.

  The simplest example is the :py:class:`pyoptmat.temperature.ConstantParameter`
  which simple returns the constant (temperature-indpendent)
  value of a single underlying parameter :code:`p`.

  The second simplest example is
  :py:class:`pyoptmat.temperature.PolynomialScaling` which takes a tensor parameter
  :code:`c` describing polynomial coefficients.  This scaling function
  evaluates the polynomial described by these coefficients at :code:`T` to
  provide the parameter temperature dependence.

  The numerical scaling functions :math:`s` default to the identity, i.e.
  no scaling.  The :py:mod:`pyoptmat.optimize` module provides some
  common, useful scaling functions, though often simply scaling by an
  appropriate order of magnitude value so that the "actual" parameters
  :math:`p` are on the order of 1.

  All of these temperature scaling rules have to be compatible with
  batched temperatures, i.e. they should expect to receive a tensor
  of shape :code:`(nbatch,)`, not just a scalar.
"""

import torch
from torch import nn


class TemperatureParameter(nn.Module):
    """
    Superclass of all temperature-dependent parameters

    This class takes care of numerical scaling on the end result, if required

    Keyword Args:
      scaling (function):   numerical scaling function, defaults to no scaling
    """

    def __init__(self, *args, scaling=lambda x: x, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaling = scaling

    def forward(self, T):
        """
        Return the actual parameter value

        Args:
          T (torch.tensor):   current temperature
        """
        return self.scaling(self.value(T))


class ConstantParameter(TemperatureParameter):
    """
    A parameter that is constant with temperature

    Args:
      pvalue (torch.tensor):    the constant parameter value

    Keyword Args:
      p_scale (function):       numerical scaling function, defaults to
                                no scaling
    """

    def __init__(self, pvalue, *args, p_scale=lambda x: x, **kwargs):
        super().__init__(*args, **kwargs)
        self.pvalue = pvalue
        self.p_scale = p_scale

    @property
    def device(self):
        """
        Return the device used by the scaling function
        """
        return self.pvalue.device

    def value(self, T):
        """
        Pretty simple, just return the value!

        Args:
          T (torch.tensor):   current temperature

        Returns:
          torch.tensor:       value at the given temperatures
        """
        return self.p_scale(self.pvalue)

    @property
    def shape(self):
        """
        The shape of the underlying parameter
        """
        return self.pvalue.shape


class ShearModulusScaling(TemperatureParameter):
    """
    Parameter that scales as:

    .. math::

      A \\mu

    where :math:`\\mu` further depends on temperature

    Args:
      A (torch.tensor): actual parameter
      mu (|TP|):        scalar, temperature-dependent shear modulus

    Keyword Args:
      A_scale (function):       numerical scaling function for A, defaults to
                                no scaling
    """

    def __init__(self, A, mu, *args, A_scale=lambda x: x, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = A
        self.mu = mu
        self.A_scale = A_scale

    @property
    def device(self):
        """
        Return the device used by the scaling function
        """
        return self.A.device

    def value(self, T):
        """
        Return the function value

        Args:
          T (torch.tensor):   current temperature

        Returns:
          torch.tensor:       value at the given temperatures
        """
        return self.A_scale(self.A) * self.mu(T)

    @property
    def shape(self):
        """
        The shape of the underlying parameter
        """
        return self.A.shape


class ShearModulusScalingExp(TemperatureParameter):
    """
    Parameter that scales as:

    .. math::

      \\exp(A) \\mu

    where :math:`\\mu` further depends on temperature

    Args:
      A (torch.tensor): actual parameter
      mu (|TP|):        scalar, temperature-dependent shear modulus

    Keyword Args:
      A_scale (function):       numerical scaling function for A, defaults to
                                no scaling
    """

    def __init__(self, A, mu, *args, A_scale=lambda x: x, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = A
        self.mu = mu
        self.A_scale = A_scale

    @property
    def device(self):
        """
        Return the device used by the scaling function
        """
        return self.A.device

    def value(self, T):
        """
        Return the function value

        Args:
          T (torch.tensor):   current temperature

        Returns:
          torch.tensor:       value at the given temperatures
        """
        return torch.exp(self.A_scale(self.A)) * self.mu(T)

    @property
    def shape(self):
        """
        The shape of the underlying parameter
        """
        return self.A.shape


class MTSScaling(TemperatureParameter):
    """
    Parameter that scales as:

    .. math::

      \\hat{\\tau}\\left\\{ 1 - \\left[ \\frac{kT}{\\mu b^3 g_0} \\right]^{1/q} \\right\\}^{1/p}

    Args:
      tau0 (torch.tensor):  threshold strength
      g0 (torch.tensor):    activation energy
      q (torch.tensor):     shape parameter
      p (torch.tensor):     shape parameter
      k (torch.tensor):     Boltzmann constant
      b (torch.tensor):     burgers vector
      mu (|TP|):            shear modulus, temperature-dependent

    Keyword Args:
      tau0_scale (function):    numerical scaling function for tau0, defaults to
                                no scaling
      g0_scale (function):      numerical scaling function for g0, defaults to
                                no scaling
      q_scale (function):       numerical scaling function for g0, defaults to
                                no scaling
      p_scale (function):       numerical scaling function for p, defaults to
                                no scaling
    """

    def __init__(
        self,
        tau0,
        g0,
        q,
        p,
        k,
        b,
        mu,
        *args,
        tau0_scale=lambda x: x,
        g0_scale=lambda x: x,
        q_scale=lambda x: x,
        p_scale=lambda x: x,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tau0 = tau0
        self.g0 = g0
        self.q = q
        self.p = p
        self.k = k
        self.b = b
        self.mu = mu

        self.tau0_scale = tau0_scale
        self.g0_scale = g0_scale
        self.q_scale = q_scale
        self.p_scale = p_scale

    @property
    def device(self):
        """
        Return the device used by the scaling function
        """
        return self.tau0.device

    def value(self, T):
        """
        Return the function value

        Args:
          T (torch.tensor):   current temperature

        Returns:
          torch.tensor:       value at the given temperatures
        """
        return self.tau0_scale(self.tau0) * (
            1
            - (self.k * T / (self.mu(T) * self.b**3.0 * self.g0_scale(self.g0)))
            ** (1 / self.q_scale(self.q))
        ) ** (1 / self.p_scale(self.p))

    @property
    def shape(self):
        """
        Shape of the underlying parameter
        """
        return self.tau0.shape


class KMRateSensitivityScaling(TemperatureParameter):
    """
    Parameter that scales as:

    .. math::

      \\frac{-\\mu b^3}{kTA}

    where :math:`\\mu` further depends on temperature

    Args:
      A (torch.tensor): Kocks-Mecking slope parameter, sets shape
      mu (|TP|):        scalar, temperature-dependent shear modulus
      b (torch.tensor): scalar, Burgers vector
      k (torch.tensor): scalar, Boltzmann constant

    Keyword Args:
      cutoff (float):       don't let n go higher than this
      A_scale (function):   numerical scaling function for A, defaults to
                            no scaling
    """

    def __init__(self, A, mu, b, k, *args, A_scale=lambda x: x, cutoff=20.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = A
        self.mu = mu
        self.b = b
        self.k = k
        self.cutoff = cutoff

        self.A_scale = A_scale

    @property
    def device(self):
        """
        Return the device used by the scaling function
        """
        return self.A.device

    def value(self, T):
        """
        Return the function value

        Args:
          T (torch.tensor):   current temperature

        Returns:
          torch.tensor:       value at the given temperatures
        """
        return torch.clip(
            -self.mu(T) * self.b**3.0 / (self.k * T * self.A_scale(self.A)),
            min=1,
            max=self.cutoff,
        )

    @property
    def shape(self):
        """
        Shape of the underlying parameter
        """
        return self.A.shape


class KMViscosityScaling(TemperatureParameter):
    """
    Parameter that varies as

    .. math::

      \\exp{B} \\mu \\dot{\\varepsilon}_0^{-1/n}

    where :math:`B` is the Kocks-Mecking intercept parameter and the
    rest are defined in the :py:class:`pyoptmat.temperature.KMRateSensitivityScaling` object.

    :math:`n` is the rate sensitivity, again given by the
    :py:class:`pyoptmat.temperature.KMRateSensitivityScaling` object

    Args:
      A (torch.tensor):     Kocks-Mecking slope parameter
      B (torch.tensor):     Kocks-Mecking intercept parameter, sets shape, must be
                            same shape as A
      mu (|TP|):            scalar, temperature-dependent shear modulus
      eps0 (torch.tensor):  scalar, reference strain rate
      b (torch.tensor):     scalar, Burger's vector
      k (torch.tensor):     scalar, Boltzmann constant

    Keyword Args:
      A_scale (function):   numerical scaling function for A, defaults to
                            no scaling
      B_scale (function):   numerical scaling function B, defaults to no
                            scaling
    """

    def __init__(
        self,
        A,
        B,
        mu,
        eps0,
        b,
        k,
        *args,
        A_scale=lambda x: x,
        B_scale=lambda x: x,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.A = A
        self.B = B
        self.mu = mu
        self.eps0 = eps0
        self.b = b
        self.k = k

        self.A_scale = A_scale
        self.B_scale = B_scale

        self.n = KMRateSensitivityScaling(
            self.A, self.mu, self.b, self.k, A_scale=self.A_scale, cutoff=1000
        )

    @property
    def device(self):
        """
        Return the device used by the scaling function
        """
        return self.A.device

    def value(self, T):
        """
        Return the function value

        Args:
          T (torch.tensor):   current temperature

        Returns:
          torch.tensor:       value at the given temperatures
        """
        n = self.n(T)
        return torch.exp(self.B_scale(self.B)) * self.mu(T) * self.eps0 ** (-1.0 / n)

    @property
    def shape(self):
        """
        Shape of the underlying parameter
        """
        return self.B.shape


class PolynomialScaling(TemperatureParameter):
    """
    Mimics np.polyval using Horner's method to evaluate a polynomial

    Args:
      coefs (torch.tensor): polynomial coefficients in the numpy convention
                            (highest order 1st)

    Keyword Args:
      coef_scale_fn (function): numerical scaling function for coefs,
                                defaults to no scaling
    """

    def __init__(self, coefs, *args, coef_scale_fn=lambda x: x, **kwargs):
        super().__init__(*args, **kwargs)
        self.coefs = coefs
        self.scale_fn = coef_scale_fn

    @property
    def device(self):
        """
        Return the device used by the scaling function
        """
        return self.coefs.device

    def value(self, T):
        """
        Return the function value

        Args:
          T (torch.tensor):   current temperature

        Returns:
          torch.tensor:       value at the given temperatures
        """
        acoefs = self.scale_fn(self.coefs)

        res = torch.zeros_like(T) + acoefs[0]
        for c in acoefs[1:]:
            res *= T
            res += c

        return res

    @property
    def shape(self):
        """
        Shape of the underlying parameter
        """
        return self.coefs[0].shape


class PiecewiseScaling(TemperatureParameter):
    """
    Piecewise linear interpolation, mimics scipy.interp1d with
    default options

    Args:
      control (torch.tensor):   temperature control points (npoints)
      values (torch.tensor):    values at control points (optional_batch, npoints)

    Keyword Args:
      values_scale_fn (function):   numerical scaling function for values,
                                    defaults to no scaling
    """

    def __init__(self, control, values, *args, values_scale_fn=lambda x: x, **kwargs):
        super().__init__(*args, **kwargs)

        self.control = control
        self.values = values

        self.values_scale_fn = values_scale_fn

    @property
    def device(self):
        """
        Return the device used by the scaling function
        """
        return self.values.device

    def value(self, T):
        """
        Return the function value

        Args:
          T (torch.tensor):   current temperature

        Returns:
          torch.tensor:       value at the given temperatures
        """
        vcurr = self.values_scale_fn(self.values)

        slopes = (vcurr[..., 1:] - vcurr[..., :-1]) / (
            self.control[..., 1:] - self.control[..., :-1]
        )

        offsets = T.unsqueeze(-1) - self.control[..., :-1]

        poss = (slopes[None, ...] * offsets + vcurr[None, ..., :-1]).reshape(
            offsets.shape
        )

        locs = torch.logical_and(
            T <= self.control[1:][(...,) + (None,) * T.dim()],
            T > self.control[:-1][(...,) + (None,) * T.dim()],
        )

        # Problem is now that we can have stuff right on control[0]...
        locs[0] = torch.logical_or(
            locs[0], T == self.control[0][(...,) + (None,) * T.dim()]
        )

        return poss[locs.movedim(0, -1)].reshape(T.shape)

    @property
    def shape(self):
        """
        Shape of the underlying parameter
        """
        return self.values.shape[1:]


class ArrheniusScaling(TemperatureParameter):
    """
    Simple Arrhenius scaling of the type

    .. math::

      A \\exp(-Q/T)

    Args:
      A (torch.tensor):  Prefactor
      Q (torch.tensor):  Activation energy (times R)

    Keyword Args
      A_scale (function):  numerical scaling for A, defaults to no scaling
      Q_scale (function):  numerical scaling for Q, defaults to no scaling
    """

    def __init__(self, A, Q, *args, A_scale=lambda x: x, Q_scale=lambda x: x, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = A
        self.Q = Q
        self.A_scale = A_scale
        self.Q_scale = Q_scale

    @property
    def device(self):
        """
        Return the device used by the scaling function
        """
        return self.A.device

    def value(self, T):
        """
        Return the function value

        Args:
          T (torch.tensor):   current temperature

        Returns:
          torch.tensor:       value at the given temperatures
        """
        return self.A_scale(self.A) * torch.exp(-self.Q_scale(self.Q) / T)

    @property
    def shape(self):
        """
        Shape of the underlying parameter
        """
        return self.A.shape


class InverseArrheniusScaling(TemperatureParameter):
    """
    Simple Arrhenius scaling of the type

    .. math::

      A (1 - \\exp(-Q/T))

    Args:
      A (torch.tensor):  Prefactor
      Q (torch.tensor):  Activation energy (times R)

    Keyword Args
      A_scale (function):  numerical scaling for A, defaults to no scaling
      Q_scale (function):  numerical scaling for Q, defaults to no scaling
    """

    def __init__(self, A, Q, *args, A_scale=lambda x: x, Q_scale=lambda x: x, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = A
        self.Q = Q
        self.A_scale = A_scale
        self.Q_scale = Q_scale

    @property
    def device(self):
        """
        Return the device used by the scaling function
        """
        return self.A.device

    def value(self, T):
        """
        Return the function value

        Args:
          T (torch.tensor):   current temperature

        Returns:
          torch.tensor:       value at the given temperatures
        """
        return self.A_scale(self.A) * (1.0 - torch.exp(-self.Q_scale(self.Q) / T))

    @property
    def shape(self):
        """
        Shape of the underlying parameter
        """
        return self.A.shape
