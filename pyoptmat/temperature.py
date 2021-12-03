"""
  Temperature interpolation formula of various kinds
"""

import torch
import torch.nn as nn

class TemperatureParameter(nn.Module):
  """
    Superclass of all temperature-dependent parameters

    This class takes care of scaling the end result, if required

    Args:
      scaling (optional):   how to scale the temperature-dependent values,
                            defaults to no scaling
  """
  def __init__(self, *args, scaling = lambda x: x, **kwargs):
    super().__init__(*args, **kwargs)
    self.scaling = scaling

  def forward(self, T):
    """
      Return the actual parameter value

      Args:
        T:      current temperature
    """
    return self.scaling(self.value(T))

class ConstantParameter(TemperatureParameter):
  """
    A parameter that is constant with temperature

    Args:
      pvalue:       the constant parameter value
  """
  def __init__(self, pvalue, *args,  p_scale = lambda x: x, **kwargs):
    super().__init__(*args, **kwargs)
    self.pvalue = pvalue
    self.p_scale = p_scale

  def value(self, T):
    """
      Pretty simple, just return the value!

      Args:
        T:          current batch temperatures
    """
    return self.p_scale(self.pvalue)

  @property
  def shape(self):
    return self.pvalue.shape

class ShearModulusScaling(TemperatureParameter):
  """
    Parameter that scales as:

      $A \\mu$

      where $\\mu$ further depends on temperature

    Args:
      A:        actual parameter
      mu:       scalar, temperature-dependent shear modulus
      
  """
  def __init__(self, A, mu, *args, A_scale = lambda x: x, **kwargs):
    super().__init__(*args, **kwargs)
    self.A = A
    self.mu = mu
    self.A_scale = A_scale

  def value(self, T):
    return self.A_scale(self.A) * self.mu(T)

  @property
  def shape(self):
    return self.A.shape

class MTSScaling(TemperatureParameter):
  """
    Parameter that scales as:
      
      $\hat{\tau}\left\{ 1 - \left[ \frac{kT}{\mu b^3 g_0} \right]^{1/q} \right\}^{1/p}

    Args:
      tau0:     threshold strength
      g0:       activation energy
      q:        shape parameter
      p:        shape parameter
      k:        Boltzmann constant
      b:        burgers vector
      mu:       shear modulus, temperature-dependent
  """
  def __init__(self, tau0, g0, q, p, k, b, mu, *args,
      tau0_scale = lambda x: x, g0_scale = lambda x: x,
      q_scale = lambda x: x, p_scale = lambda x: x, **kwargs):
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

  def value(self, T):
    return self.tau0_scale(self.tau0) * (1 - 
        (self.k*T/(self.mu(T) * self.b**3.0 * 
      self.g0_scale(self.g0)))**(1/self.q_scale(self.q)))**(1/self.p_scale(self.p))

  @property
  def shape(self):
    return self.tau0.shape

class KMRateSensitivityScaling(TemperatureParameter):
  """
    Parameter that scales as:

      $\\frac{-\\mu b^3}{kTA}$

    where $\\mu$ further depends on temperature

    Args:
      A:        Kocks-Mecking slope parameter, sets shape
      mu:       scalar, temperature-dependent shear modulus
      b:        scalar, Burgers vector
      k:        scalar, Boltzmann constant

    Additional args:
      cutoff:   don't let n go higher than this
  """
  def __init__(self, A, mu, b, k, cutoff = 20.0,
      *args, A_scale = lambda x: x, **kwargs):
    super().__init__(*args, **kwargs)
    self.A = A
    self.mu = mu
    self.b = b
    self.k = k
    self.cutoff = cutoff

    self.A_scale = A_scale

  def value(self, T):
    """
      Actual temperature-dependent value

      Args:
        T:      current temperatures
    """
    return torch.clip(
        -self.mu(T) * self.b**3.0 / (self.k * T * self.A_scale(self.A)), 
        min = 1, max = self.cutoff)

  @property
  def shape(self):
    return self.A.shape

class KMViscosityScaling(TemperatureParameter):
  """
    Parameter that varies as

      $\exp{B} \mu \dot{\varepsilon}_0^{-1/n}

    where $B$ is the Kocks-Mecking intercept parameter and the
    rest are defined in the `KMRateSensitivityScaling` object.
    
    $n$ is the rate sensitivity, again given by the `KMRateSensitivityScaling`
    object

    Args:
      A:        Kocks-Mecking slope parameter
      B:        Kocks-Mecking intercept parameter, sets shape, must be
                same shape as A
      mu:       scalar, temperature-dependent shear modulus
      eps0:     scalar, reference strain rate
      b:        scalar, Burger's vector
      k:        scalar, Boltzmann constant
  """
  def __init__(self, A, B, mu, eps0, b, k, *args, A_scale = lambda x: x,
      B_scale = lambda x: x, **kwargs):
    super().__init__(*args, **kwargs)
    self.A = A
    self.B = B
    self.mu = mu
    self.eps0 = eps0
    self.b = b
    self.k = k

    self.A_scale = A_scale
    self.B_scale = B_scale

    self.n = KMRateSensitivityScaling(self.A, self.mu, self.b, self.k,
        A_scale = self.A_scale)

  def value(self, T):
    """
      Actual temperature-dependent value

      Args:
        T:      current temperatures
    """
    n = self.n(T)
    return torch.exp(self.B_scale(self.B)) * self.mu(T) * self.eps0**(-1.0/n)

  @property
  def shape(self):
    return self.B.shape

class KMViscosityScalingCutoff(TemperatureParameter):
  """
    Parameter that varies as

      $\exp{B} \mu \dot{\varepsilon}_0^{-1/n}

    where $B$ is the Kocks-Mecking intercept parameter and the
    rest are defined in the `KMRateSensitivityScaling` object.

    This variant cuts off the viscosity in the high rate/low temperature 
    regime to limit it by C * \mu 
    
    $n$ is the rate sensitivity, again given by the `KMRateSensitivityScaling`
    object

    Args:
      A:        Kocks-Mecking slope parameter
      B:        Kocks-Mecking intercept parameter, sets shape, must be
                same shape as A
      C:        High rate/low temperature cutff
      mu:       scalar, temperature-dependent shear modulus
      eps0:     scalar, reference strain rate
      b:        scalar, Burger's vector
      k:        scalar, Boltzmann constant
  """
  def __init__(self, A, B, C, mu, eps0, b, k, *args, A_scale = lambda x: x,
      B_scale = lambda x: x, C_scale = lambda x: x, **kwargs):
    super().__init__(*args, **kwargs)
    self.A = A
    self.B = B
    self.C = C
    self.mu = mu
    self.eps0 = eps0
    self.b = b
    self.k = k

    self.A_scale = A_scale
    self.B_scale = B_scale
    self.C_scale = C_scale

    self.n = KMRateSensitivityScaling(self.A, self.mu, self.b, self.k,
        A_scale = self.A_scale, cutoff = 1000)

  def value(self, T):
    """
      Actual temperature-dependent value

      Args:
        T:      current temperatures
    """
    n = self.n(T)
    return torch.minimum(
        torch.exp(self.B_scale(self.B)) * self.mu(T) * self.eps0**(-1.0/n),
        torch.exp(self.C_scale(self.C)) * self.mu(T))

  @property
  def shape(self):
    return self.B.shape

class PolynomialScaling(TemperatureParameter):
  """
    Mimics np.polyval using Horner's method to evaluate a polynomial

    Args:
      coefs:        polynomial coefficients in the numpy convention (highest order 1st)
  """
  def __init__(self, coefs, *args, coef_scale_fn = lambda x: x, **kwargs):
    super().__init__(*args, **kwargs)
    self.coefs = coefs
    self.scale_fn = coef_scale_fn

  def value(self, T):
    acoefs = self.scale_fn(self.coefs)

    res = torch.zeros_like(T) + acoefs[0]
    for c in acoefs[1:]:
      res *= T
      res += c

    return res

  @property
  def shape(self):
    return self.coefs[0].shape
