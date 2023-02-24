#!/usr/bin/env python3

"""
Sample problem trying to infer the response of a chain of serially
connected mass-spring-dashpot 1D elements, fixed at one end of the
chain and subject to a force on the other end.

The number of elements in the chain is `n_chain`.

This example
1. Generates `n_samples` trajectories under a random force described by
   ...  We separate out only the end displacements as observable,
   all the interior displacements are treated as hidden state.
2. Try to infer the mass/spring/dashpot constants and the system response
   using the analytical ODE as a model, given the end-displacement
   trajectories as input.
3. Again try to infer the response from the end displacement trajectories
   but now using a neural ODE as the model.

The integration will use `n_time` points in the integration.
"""

import torch
import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append("../..")
from pyoptmat import ode, utility

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# I'm assuming single precision will be fine for this, but here is 
# where you would tell it to use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

class MassDamperSpring(torch.nn.Module):
    """
    Mass, spring, damper system of arbitrary size

    I use simple substitution to solve for the velocity and displacement
    in a first order system, so the size of the system is twice the size of the
    number of elements.

    Args:
        K (torch.tensor): (n_chain,) vector of stiffnesses
        C (torch.tensor): (n_chain,) vector of dampers
        M (torch.tensor): (n_chain,) vector of masses
        force (function): gives f(t)
    """
    def __init__(self, K, C, M, force):
        super().__init__()

        self.K = torch.nn.Parameter(K)
        self.C = torch.nn.Parameter(C)
        self.M = torch.nn.Parameter(M)
        self.force = force
        
        self.half_size = K.shape[0]
        self.size = 2 * self.half_size

    def forward(self, t, y):
        """
        Return the state rate

        Args:
            t (torch.tensor): (nchunk, nbatch) tensor of times
            y (torch.tensor): (nchunk, nbatch, size) tensor of state

        Returns:
            y_dot: (nchunk, nbatch, size) tensor of state rates
            J:     (nchunk, nbatch, size, size) tensor of Jacobians
        """
        ydot = torch.zeros_like(y)
        J = torch.zeros(y.shape + (self.size,), device = t.device)

        # Separate out for convenience
        u = y[...,:self.half_size]
        v = y[...,self.half_size:]
        
        # Differences
        du = torch.zeros_like(u)
        du[...,:-1] = torch.diff(u, dim = -1)
        du[...,-1] = -u[...,-1]
        dv = torch.zeros_like(v)
        dv[...,:-1] = torch.diff(v, dim = -1)
        dv[...,-1] = -v[...,-1]
        
        # Rate
        ydot[...,:self.half_size] = v
        ydot[...,self.half_size] = (self.force(t) - self.C[...,0] * dv[...,0] - self.K[...,0] * du[...,0]) / self.M[...,0]
        ydot[...,self.half_size+1:] = (
                - self.C[...,1:] * dv[...,1:]
                - self.K[...,1:] * du[...,1:]
                + self.C[...,:-1] * dv[...,:-1]
                + self.K[...,:-1] * du[...,:-1]
                ) / self.M[...,1:]

        # Derivative
        J[...,:self.half_size,self.half_size:] = torch.eye(self.half_size, device = t.device)
        J[...,self.half_size:,:self.half_size] = torch.diag_embed(self.K / self.M) - torch.diag_embed(self.K[:-1] / self.M[:-1], offset = 1)
        J[...,self.half_size:,self.half_size:] = torch.diag_embed(self.C / self.M) - torch.diag_embed(self.C[:-1] / self.M[:-1], offset = 1)

        return ydot, J

    def initial_condition(self, nsamples):
        """
        I don't want to learn the initial condition (we could) 
        so just return deterministic zeros
        """
        return torch.zeros(nsamples, self.size, device = device)

class NeuralODE(torch.nn.Module):
    """
    Neural ODE model:
    - input and output of size 1 + n_hidden
    - n_layers linear layers of size n_inter
    """

def random_walk(time, mean, scale, mag):
    """
    Simulate a random walk with velocity of the given mean and scale
    """
    results = torch.zeros_like(time)
    for i in range(1, time.shape[0]):
        dt = time[i] - time[i-1]
        v = mag*torch.normal(mean, scale)
        results[i] = results[i-1] + v * dt

    return results

if __name__ == "__main__":
    # Basic parameters
    n_chain = 4     # Number of spring-dashpot-mass elements
    n_samples = 1   # Number of random force samples
    n_time = 500    # Number of time steps
    integration_method = 'backward-euler'

    # Time chunking -- best value may vary on your system
    n_chunk = 1

    # Ending time
    t_end = 1.0

    # Mean and standard deviation for random force "velocity"
    force_mean = torch.tensor(0.0, device = device)
    force_std = torch.tensor(1.0, device = device)
    force_mag = torch.tensor(10.0, device = device)

    # True values of the mass, damping, and stiffness
    K = torch.rand(n_chain, device = device)
    C = torch.rand(n_chain, device = device)
    M = torch.rand(n_chain, device = device)

    # Time values
    time = torch.linspace(0, t_end, n_time, device = device).unsqueeze(-1).expand(n_time, n_samples)

    # 1. Generate the data

    # Generate the force values
    force = random_walk(time, force_mean.unsqueeze(0).expand(n_samples), force_std.unsqueeze(0).expand(n_samples), force_mag.unsqueeze(0).expand(n_samples))

    # Plot them
    plt.figure()
    plt.plot(time.cpu().numpy(), force.cpu().numpy())
    plt.show()

    # Interpolate in time
    force_continuous = utility.ArbitraryBatchTimeSeriesInterpolator(time, force)

    # Setup the ground truth model
    model = MassDamperSpring(K, C, M, force_continuous)

    # Do a quick finite difference check for the Jacobian
    test_state = torch.rand(n_chunk, n_samples, model.size, device = device)
    _, Jtest1 = model(time[0:n_chunk], test_state)
    Jtest2 = utility.batch_differentiate(lambda x: model(time[0:n_chunk], x)[0], test_state,
            nbatch_dim = 2)

    print(Jtest1[0,0])
    print(Jtest2[0,0])

    print(torch.max(Jtest1-Jtest2))

    sys.exit()

    # Generate the initial condition
    y0 = model.initial_condition(n_samples)

    # Generate the data
    with torch.no_grad():
        y_data = ode.odeint(model, y0, time, method = integration_method, block_size = n_chunk)

    # The observations will just be the first entry
    observable = y_data[...,0]
    
    # Plot them
    plt.figure()
    plt.plot(time.cpu().numpy(), observable.cpu().numpy())
    plt.show()

    sys.exit()

    # 2. Infer with the actual model
    ode_model = MassDamperSpring(torch.rand(n_chain, device = device),
            torch.rand(n_chain, device = device),
            torch.rand(n_chain, device = device), force_continuous)
    
    # Training parameters...
    niter = 1000
    lr = 1.0e-3
    loss = torch.nn.MSELoss(reduction = "sum")

    # Optimization setup
    optim = torch.optim.Adam(ode_model.parameters(), lr)
    def closure():
        optim.zero_grad()
        pred = ode.odeint_adjoint(ode_model, y0, time, method = integration_method, block_size = n_chunk)
        obs = pred[...,0]
        lossv = loss(obs, observable)
        lossv.backward()
        return lossv

    # Optimization loop
    t = tqdm.tqdm(range(niter), total = niter, desc = "Loss:     ")
    loss_history = []
    for i in t:
        closs = optim.step(closure)
        loss_history.append(closs.detach().cpu().numpy())
        t.set_description("Loss: %3.2e" % loss_history[-1])

    # Plot the loss history
    plt.figure()
    plt.plot(loss_history)
    plt.show()

    # Make the final predictions
    with torch.no_grad():
        ode_preds = ode.odeint(ode_model, y0, time, method = integration_method, block_size = n_chunk)
        ode_obs = ode_preds[...,0]

    # Make a plot
    plt.figure()
    for i in range(n_samples):
        l, = plt.plot(time.cpu().numpy()[:,i], observable.cpu().numpy()[:,i])
        plt.plot(time.cpu().numpy()[:,i], ode_obs.cpu().numpy()[:,i], ls = '--', color = l.get_color())
    plt.show()

    # 2. Infer with a neural ODE

