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
  def __init__(self, pvalue, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.pvalue = pvalue

  def value(self, T):
    """
      Pretty simple, just return the value!

      Args:
        T:          current batch temperatures
    """
    return self.pvalue

  @property
  def shape(self):
    return self.pvalue.shape
