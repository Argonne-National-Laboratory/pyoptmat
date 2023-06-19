#!/usr/bin/env python3

import sys
sys.path.append('../../..')

import math

import xarray as xr
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import tqdm

from pyoptmat import optimize, experiments, models, flowrules, hardening, temperature, scaling

from functorch import jacrev, jacfwd, vmap

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

class NeuralInelasticModel(nn.Module):
    """
    A simple neural ODE with nlayers fully connected internal and ninternal internal variables
    Use reLU activation for now

    The network should account for (stress, internal_state, strain_rate, T) = 3 + nstate inputs
    and have (stress, internal_state) = 1 + nstate outputs

    Args:
        w1 ... w3: network weights
        b1 ... b3: network biases
    """
    def __init__(self, w1, w2, w3, b1, b2, b3, **kwargs):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

        ifeatures = self.w1.shape[-1]
        ofeatures = self.w3.shape[-2]

        if ifeatures < 3:
            raise ValueError("Input feature size must be greater than 3")
        if ofeatures + 2 != ifeatures:
            raise ValueError("Output feature size must be 2 less than the input feature size")


        # Infer the number of internal variables
        self.ninternal = ofeatures - 1

        # Activation
        self.activation = nn.ReLU()

    def rate(self, t, y, erate, T):
        """
        Evaluate the current state rate

        Args:
          t:              (...,) times
          y:              (...,nsize) [stress, history]
          erate:          (...,) strain rates
          T:              (...,) temperatures
        """
        x = torch.cat([y, erate.unsqueeze(-1), T.unsqueeze(-1)], dim = -1)
        
        v1 = self.activation(torch.einsum('...ij,...j',self.w1,x)+self.b1) 
        v2 = self.activation(torch.einsum('...ij,...j',self.w2,v1)+self.b2) 
        v3 = torch.einsum('...ij,...j',self.w3,v2)+self.b3

        return v3

    def forward(self, t, y, erate, T):
        """
        Evaluate the full model, including Jacobians

        Args:
          t:              (...,) times
          y:              (...,nsize) [stress, history]
          erate:          (...,) strain rates
          T:              (...,) temperatures

        Returns:
          y_dot:          (...,nsize) state rate
          d_y_dot_d_y:    (...,nsize,nsize) Jacobian wrt the state
          d_y_dot_d_erate:(...,nsize) Jacobian wrt the strain rate
          d_y_dot_d_T:    (...,nsize) derivative wrt temperature (unused)
        """
        ydot = self.rate(t, y, erate, T)
        dydot_dy = vmap(vmap(jacfwd(self.rate, argnums = 1)))(t, y, erate, T)
        dydot_de = vmap(vmap(jacfwd(self.rate, argnums = 2)))(t, y, erate, T)
        dydot_dT = vmap(vmap(jacfwd(self.rate, argnums = 3)))(t, y, erate, T)

        return ydot, dydot_dy, dydot_de, dydot_dT

    @property
    def nsize(self):
        return 1 + self.ninternal 

if __name__ == "__main__":
    # Time chunking
    time_chunk_size = 5

    # Load in the data
    input_data = xr.open_dataset("data.nc")
    data, results, cycles, types, control = experiments.load_results(
            input_data, device = device)

    # Scale the data and results -- this tends to help NN models converge
    for i in range(3):
        data[i] = (data[i] - torch.min(data[i])) / (torch.max(data[i]) - torch.min(data[i]))

    results = (results - torch.min(results)) / (torch.max(results) - torch.min(results))

    # Model making function
    def make(*args, **kwargs):
        return models.ModelIntegrator(NeuralInelasticModel(*args), **kwargs).to(device)
    
    # Setup initial guesses
    ninternal = 4
    nfeatures = ninternal + 1
    names = ["w1", "w2", "w3", "b1", "b2", "b3"]
    init_dist_weight = torch.distributions.uniform.Uniform(-1/math.sqrt(float(nfeatures)), 1/math.sqrt(float(nfeatures)))
    init_dist_bias = torch.distributions.uniform.Uniform(-1/math.sqrt(float(nfeatures)), 1/math.sqrt(float(nfeatures)))
    initial_values = [
            init_dist_weight.sample((nfeatures, nfeatures+2)).to(device),
            init_dist_weight.sample((nfeatures,nfeatures)).to(device),
            init_dist_weight.sample((nfeatures,nfeatures)).to(device),
            init_dist_bias.sample((nfeatures,)).to(device),
            init_dist_bias.sample((nfeatures,)).to(device),
            init_dist_bias.sample((nfeatures,)).to(device)
            ]


    # Setup the model
    model = optimize.DeterministicModel(lambda *args, **kwargs: make(*args, 
        block_size = time_chunk_size, **kwargs),
        names, initial_values)
    
    # Check what our initial values give us...
    with torch.no_grad():
        test_results = model(data.to(device), cycles.to(device), types.to(device),
                control.to(device))
    plt.plot(data[-1].cpu().numpy(), test_results.cpu().numpy())
    plt.show()

    # Setup the optimizer and loss
    lr = 1.0e-3
    niter = 500
    num_samples = 1

    optim = torch.optim.Adam(model.parameters(), lr = lr)

    loss = torch.nn.MSELoss(reduction="sum") 

    # Optimization closure
    def closure():
        optim.zero_grad()
        pred = model(data, cycles, types, control)
        lossv = loss(pred, results)
        lossv.backward()
        return lossv
   
    # Actually do the optimization!
    with torch.autograd.set_detect_anomaly(True):
        t = tqdm.tqdm(range(niter), total=niter, desc="Loss:    ")
        loss_history = []
        for i in t:
            closs = optim.step(closure)
            loss_history.append(closs.detach().cpu().numpy())
            t.set_description("Loss: %3.2e" % loss_history[-1])
