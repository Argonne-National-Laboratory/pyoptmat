"""Helpers for neural ODEs

Contains miscellaneous helper classes to make working with neural ODEs
easier in pyoptmat.
"""

import torch

from pyoptmat import utility


class NeuralODE(torch.nn.Module):
    """Simple wrapper for simple forced neural ODEs

    Mathematically these define a model of the type

    .. math::

        \\dot{y} = m(f(t), y)

    where :math:`y` is the state and :math:`f` is some forcing
    function.

    The neural model is :math:`m` here, which takes a concatenated
    combination of :math:`y` and :math:`f(t)` as input and outputs
    :math:`\\dot{y}`.

    If dim(y) is n and dim(f(t)) is s then the neural network must
    take n+s inputs and produce n outputs.

    This class handles arbitrary leading batch dimensions and also
    using torch AD to calculate the (batched) Jacobian for implicit
    integration schemes.

    Args:
        model (callable): neural function
        force (callable): forcing function
    """

    def __init__(self, model, force, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model
        self.force = force

        # Wrapped function to calculate jacobian
        @utility.jacobianize()
        def model_and_jacobian(x):
            return self.model(x)

        self.m_and_j = model_and_jacobian

    def forward(self, t, y):
        """Forward call returns rate and Jacobian

        Args:
            t (torch.tensor): current times
            y (torch.tensor): current state
        """
        f = self.force(t)
        x = torch.cat((y, f), dim=-1)

        val, jac = self.m_and_j(x)

        return val, jac[0][..., : -f.shape[-1]]
