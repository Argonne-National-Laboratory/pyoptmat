#!/usr/bin/env python3

"""
    Simple helper to make a plot illustrating the variation in the
    synthetic experimental data.
"""

import sys

sys.path.append("../../..")

import xarray as xr
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pyoptmat import temperature, experiments

from maker import make_model, downsample

# Various fixed parameters for the KM model
eps0 = 1e6
k = 1.380649e-20
b = 2.02e-7
eps0_ri = 1e-10

# Constant temperature to run simulations at 
T = 550.0 + 273.15

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

def find_yield(strain, stress, offset = 0.2/100):
    E = (stress[1] - stress[0]) / (strain[1] - strain[0])
    S = E * (strain - offset)
    i = torch.argmax(stress-S, dim = 0, keepdim = True) + 1
    return torch.gather(stress, 0, i).squeeze(0)

if __name__ == "__main__":
    scale = 0.01

    data = xr.load_dataset("scale-%3.2f.nc" % scale)

    strain = torch.tensor(data.strain.data.reshape(-1, data.nrates, data.nsamples))
    stress = torch.tensor(data.stress.data.reshape(-1, data.nrates, data.nsamples))
    erates = torch.logspace(-2, -8, 7)
    erates = torch.repeat_interleave(erates.unsqueeze(-1), 50, dim = 1)

    mu = temperature.PolynomialScaling(
        [-2.60610204e-05, 3.61911162e-02, -4.23765368e01, 8.44212545e04]
    )
    muv = mu(torch.tensor(T))

    g = k* T / (muv * b**3.0) * torch.log(eps0 / erates)
    Sy = find_yield(strain, stress)
    muSy = Sy / muv
    
    model = make_model(torch.tensor(0.59), torch.tensor(-3.08), torch.tensor(-5.88),
            torch.tensor(199.88), torch.tensor(4.70))

    scale = 0.01
    nsamples = 50  # at each strain rate
    inp_data, results, cycles, types, control = downsample(
        experiments.load_results(data),
        nsamples,
        data.nrates,
        data.nsamples,
    )
    
    with torch.no_grad():
        results = model.solve_strain(inp_data[0], inp_data[2], inp_data[1])
    stress = results[:,:,0]
    Sy_res = find_yield(inp_data[2], stress)
    
    Sy_res = Sy_res.reshape(data.nrates,data.nsamples)
    
    gs = torch.argsort(g.flatten())
    g1 = g.flatten()[gs]
    y1 = muSy.flatten()[gs]

    sns.lineplot(x = g1, y = y1, errorbar = ('ci', 95))
    sns.scatterplot(x = g.flatten(), y = muSy.flatten())

    sns.lineplot(x = g[:,0], y = Sy_res[:,0] / muv)
    plt.yscale('log')
    plt.xlabel(r"$\frac{kT}{\mu b^3} \log{\frac{\dot{\varepsilon}_0}{\dot{\varepsilon}}}$")
    plt.ylabel(r"$\frac{\sigma_y}{\mu}$")
    plt.legend(loc='best', labels = ['Regression, mean', 'Regression, 95%',  'Data', 'Model'])
    plt.tight_layout()
    plt.savefig("km-nice.png", dpi = 300)

