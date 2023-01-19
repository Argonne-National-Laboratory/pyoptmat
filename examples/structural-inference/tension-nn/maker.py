#!/usr/bin/env python3

"""
    Helper functions for the structural material model inference with
    tension tests examples.
"""

import sys

sys.path.append("../../..")

import numpy as np
import numpy.random as ra

import xarray as xr

import torch
import torch.nn as nn
from functorch import jacrev, vmap
from pyoptmat import models, flowrules, hardening, optimize
from pyoptmat.temperature import ConstantParameter as CP

from tqdm import tqdm

import tqdm

import warnings

warnings.filterwarnings("ignore")

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Actual parameters
E_true = 150000.0
R_true = 200.0
d_true = 5.0
n_true = 7.0
eta_true = 300.0
s0_true = 50.0

# Scale factor used in the model definition
sf = 0.5

nsize = 10
nscale = torch.tensor(1.0e-5)
bscale = torch.tensor(0.1)

class Lin(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.weight = weight
        self.bias = bias

        if self.weight.dim() == 3:
            self.squeeze = True
        else:
            self.squeeze = False

    def forward(self, x):
        if self.squeeze:
            return self.weight.bmm(x.unsqueeze(-1)).squeeze(-1) + self.bias
        else:
            return x.matmul(self.weight.t()) + self.bias

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * x

class Sig(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)

class Abs(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)

class Div(nn.Module):
    def __init__(self, val):
        super().__init__()
        self.val = val

    def forward(self, x):
        return 1.0/(x+self.val)


class NNIsotropicHardeningModel(hardening.IsotropicHardeningModel):
    """
    Voce isotropic hardening, defined by

    .. math::

      \\sigma_{iso} = h

      \\dot{h} = d (R - h) \\left|\\dot{\\varepsilon}_{in}\\right|

    Args:
      R (|TP|): saturated increase/decrease in flow stress
      d (|TP|): parameter controlling the rate of saturation
    """

    def __init__(self, weights1, biases1, weights2, biases2, weights3, biases3):
        super().__init__()

        self.model = torch.nn.Sequential(
                Scale(nscale),
                Div(1.0e-3),
                Lin(weights1, biases1),
                nn.ReLU(),
                Lin(weights2, biases2),
                nn.ReLU(),
                Lin(weights3, biases3),
                Abs(),
                Scale(bscale))
       
    def value(self, h):
        """
        Map from the vector of internal variables to the isotropic hardening
        value

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the isotropic hardening value
        """
        return h[:, 0]

    def dvalue(self, h):
        """
        Derivative of the map with respect to the internal variables

        Args:
          h (torch.tensor):   the vector of internal variables for this model

        Returns:
          torch.tensor:       the derivative of the isotropic hardening value
                              with respect to the internal variables
        """
        return torch.ones((h.shape[0], 1), device=h.device)

    @property
    def nhist(self):
        """
        The number of internal variables: here just 1
        """
        return 1

    def model_value(self, x):
        return self.model(x)

    def model_derivative(self, x):
        return jacrev(self.model)(x).diagonal(dim1=0,dim2=2).transpose(0,-1)

    def history_rate(self, s, h, t, ep, T, e):
        """
        The rate evolving the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       internal variable rate
        """
        return self.model_value(h) * torch.abs(ep).unsqueeze(-1)

    def dhistory_rate_dstress(self, s, h, t, ep, T, e):
        """
        The derivative of this history rate with respect to the stress

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to stress
        """
        return torch.zeros_like(h)

    def dhistory_rate_dhistory(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the internal variables

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to history
        """
        return self.model_derivative(h) * torch.abs(ep[:,None,None])

    def dhistory_rate_derate(self, s, h, t, ep, T, e):
        """
        The derivative of the history rate with respect to the inelastic
        strain rate

        Args:
          s (torch.tensor):   stress
          h (torch.tensor):   history
          t (torch.tensor):   time
          ep (torch.tensor):  the inelastic strain rate
          T (torch.tensor):   the temperature
          e (torch.tensor):   total strain rate

        Returns:
          torch.tensor:       derivative with respect to the inelastic rate
        """
        return torch.unsqueeze(
            self.model_value(h) * torch.sign(ep).unsqueeze(-1), 1
        )

def make_model(E, n, eta, s0, weights1, biases1, weights2, biases2, weights3, biases3, device=torch.device("cpu"), **kwargs):
    """
    Key function for the entire problem: given parameters generate the model
    """
    isotropic = NNIsotropicHardeningModel(weights1, biases1, weights2, biases2, weights3, biases3)
    kinematic = hardening.NoKinematicHardeningModel()
    flowrule = flowrules.IsoKinViscoplasticity(
        CP(
            n,
            scaling=optimize.bounded_scale_function(
                (
                    torch.tensor(n_true * (1 - sf), device=device),
                    torch.tensor(n_true * (1 + sf), device=device),
                )
            ),
        ),
        CP(
            eta,
            scaling=optimize.bounded_scale_function(
                (
                    torch.tensor(eta_true * (1 - sf), device=device),
                    torch.tensor(eta_true * (1 + sf), device=device),
                )
            ),
        ),
        CP(
            s0,
            scaling=optimize.bounded_scale_function(
                (
                    torch.tensor(s0_true * (1 - sf), device=device),
                    torch.tensor(s0_true * (1 + sf), device=device),
                )
            ),
        ),
        isotropic,
        kinematic,
    )
    model = models.InelasticModel(
        CP(
            E,
            scaling=optimize.bounded_scale_function(
                (
                    torch.tensor(E_true * (1 - sf), device=device),
                    torch.tensor(E_true * (1 + sf), device=device),
                )
            ),
        ),
        flowrule,
    )

    return models.ModelIntegrator(model, **kwargs)


def generate_input(erates, emax, ntime):
    """
    Generate the times and strains given the strain rates, maximum strain, and number of time steps
    """
    strain = torch.repeat_interleave(
        torch.linspace(0, emax, ntime, device=device)[None, :], len(erates), 0
    ).T.to(device)
    time = strain / erates

    return time, strain


def downsample(rawdata, nkeep, nrates, nsamples):
    """
    Return fewer than the whole number of samples for each strain rate
    """
    ntime = rawdata[0].shape[1]
    return tuple(
        data.reshape(data.shape[:-1] + (nrates, nsamples))[..., :nkeep].reshape(
            data.shape[:-1] + (-1,)
        )
        for data in rawdata
    )


if __name__ == "__main__":
    # Running this script will regenerate the data
    ntime = 200
    emax = 0.5
    erates = torch.logspace(-2, -8, 7, device=device)
    nrates = len(erates)
    nsamples = 50

    scales = [0.0, 0.01, 0.05, 0.1, 0.15]

    times, strains = generate_input(erates, emax, ntime)

    for scale in scales:
        print("Generating data for scale = %3.2f" % scale)
        full_times = torch.empty((ntime, nrates, nsamples), device=device)
        full_strains = torch.empty_like(full_times)
        full_stresses = torch.empty_like(full_times)
        full_temperatures = torch.zeros_like(full_strains)

        for i in tqdm.tqdm(range(nsamples)):
            full_times[:, :, i] = times
            full_strains[:, :, i] = strains

            # True values are 0.5 with our scaling so this is easy
            model = make_model(
                torch.tensor(0.5, device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
                torch.tensor(ra.normal(0.5, scale), device=device),
            )

            with torch.no_grad():
                full_stresses[:, :, i] = model.solve_strain(
                    times, strains, full_temperatures[:, :, i]
                )[:, :, 0]

        full_cycles = torch.zeros_like(full_times, dtype=int, device=device)
        types = np.array(["tensile"] * (nsamples * len(erates)))
        controls = np.array(["strain"] * (nsamples * len(erates)))

        ds = xr.Dataset(
            {
                "time": (["ntime", "nexp"], full_times.flatten(-2, -1).cpu().numpy()),
                "strain": (
                    ["ntime", "nexp"],
                    full_strains.flatten(-2, -1).cpu().numpy(),
                ),
                "stress": (
                    ["ntime", "nexp"],
                    full_stresses.flatten(-2, -1).cpu().numpy(),
                ),
                "temperature": (
                    ["ntime", "nexp"],
                    full_temperatures.cpu().flatten(-2, -1).numpy(),
                ),
                "cycle": (["ntime", "nexp"], full_cycles.flatten(-2, -1).cpu().numpy()),
                "type": (["nexp"], types),
                "control": (["nexp"], controls),
            },
            attrs={"scale": scale, "nrates": nrates, "nsamples": nsamples},
        )

        ds.to_netcdf("scale-%3.2f.nc" % scale)
