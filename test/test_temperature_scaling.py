import unittest

import numpy as np
import torch

from pyoptmat import temperature

class TestConstantParameter(unittest.TestCase):
  def test_value(self):
    pshould = torch.tensor([1.0,2.0])
    obj = temperature.ConstantParameter(pshould)
    pval = obj(torch.tensor(1.0))

    self.assertTrue(np.allclose(pshould.numpy(), pval.numpy()))
