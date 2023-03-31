#!/usr/bin/env python

"""
    This example integrates a more complicated system of ODEs:
    a model of coupled neurons responding to a cyclic 
    electrical current (the Hodgkin and Huxley model, as
    described by :cite:`schwemmer2012theory`).

    The number of neurons, current cycles, time steps per 
    cycle, vectorized time steps, etc. can be customized.
    This is a decent benchmark problem for examining the performance
    of pyoptmat on your machine.
"""

import sys
sys.path.append('../..')

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from pyoptmat import ode, experiments, utility
import time

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def driving_current(I_max, R, rate, thold, chold, Ncycles, nload,
        nhold, neq):
    """
        Setup cyclic current histories

        Args:
            I_max (tuple): (min,max) uniform distribution of max currents
            R (tuple): (min,max) uniform distribution of load ratios
            rate (tuple): (min, max) uniform distribution of current rates
            thold (tuple): (min, max) uniform distribution of hold at max current times
            chold (tuple): (min, max) uniform distribution of hold at min current times
            Ncycles (int): number of load cycles
            nload (int): number of time steps for current ramp
            nhold (int): number of time steps for current holds
            neq (int): number of independent current histories
    """
    Ntotal = (4*nload + 2*nhold) * Ncycles + 1
    times = torch.empty((Ntotal,neq))
    I = torch.empty((Ntotal,neq))
    for i in range(neq):
        cycle = experiments.generate_random_cycle(
                I_max, R, rate, thold, chold)
        ti, Ii, ci = experiments.sample_cycle_normalized_times(
                cycle, Ncycles, nload, nhold)
        times[:,i] = torch.tensor(ti)
        I[:,i] = torch.tensor(Ii)
    
    return utility.ArbitraryBatchTimeSeriesInterpolator(
            times.to(device), I.to(device)), times.to(device), I.to(device)

class HodgkinHuxleyCoupledNeurons(torch.nn.Module):
    """
        As described by :cite:`schwemmer2012theory`

        .. math::

            F(y) = \\begin{bmatrix} F_1(y_1)
            \\\\ F_2(y_2)
            \\\\ \\vdots
            \\\\ F_n(y_n) 
            \\end{bmatrix}

        with

        .. math::

            y = \\begin{bmatrix} y_1
            \\\ y_2
            \\\ \\vdots
            \\\ y_n
            \\end{bmatrix}

        and

        .. math::

            F_i(y_i) = \\begin{bmatrix} V_i
            \\\\ m_i
            \\\\ h_i
            \\\\ n_i
            \\end{bmatrix}

        and

        .. math::

            F_i(y_i) = \\begin{bmatrix} \\frac{1}{C_i}(-g_{Na,i}m_i^3h_i(V_i-E_{Na,i})-g_{K,i}n_i^4(V_i-E_{k,i})-g_{L,i}(V_i-E_{L,i})+I(t) + \\Delta V_{ij})
            \\\\ \\frac{m_{\\infty,i}-m_i}{\\tau_{m,i}}
            \\\\ \\frac{h_{\\infty,i}-h_i}{\\tau_{h,i}}
            \\\\ \\frac{n_{\\infty,i}-n_i}{\\tau_{n,i}}
            \\end{bmatrix}

        with

        .. math::

            \\Delta V_{ij} = \\sum_{j=1}^{n_{neurons}}\\frac{g_{C,i}(V_i-V_j)}{C_i}

    """
    def __init__(self, C, g_Na, E_Na, g_K, E_K, g_L, E_L, m_inf,
            tau_m, h_inf, tau_h, n_inf, tau_n, g_C, I):
        super().__init__()
        self.C = nn.Parameter(C)
        self.g_Na = nn.Parameter(g_Na)
        self.E_Na = nn.Parameter(E_Na)
        self.g_K = nn.Parameter(g_K)
        self.E_K = nn.Parameter(E_K)
        self.g_L = nn.Parameter(g_L)
        self.E_L = nn.Parameter(E_L)
        self.m_inf = nn.Parameter(m_inf)
        self.tau_m = nn.Parameter(tau_m)
        self.h_inf = nn.Parameter(h_inf)
        self.tau_h = nn.Parameter(tau_h)
        self.n_inf = nn.Parameter(n_inf)
        self.tau_n = nn.Parameter(tau_n)
        self.g_C = nn.Parameter(g_C)

        self.I = I

        self.n_neurons = self.g_C.shape[0]
        self.n_equations = self.n_neurons * 4

    def forward(self, t, y):
        """
            Evaluate the system of ODEs at a particular state

            Args:
                t (torch.tensor): current time 
                y (torch.tensor): current state

            Returns:
                y_dot (torch.tensor): current ODEs
                J (torch.tensor): current jacobian
        """
        Ic = self.I(t)

        V = y[...,0::4].clone()
        m = y[...,1::4].clone()
        h = y[...,2::4].clone()
        n = y[...,3::4].clone()

        ydot = torch.zeros_like(y)
        
        # V
        ydot[...,0::4] = 1.0 / self.C[None,...] * (
                -self.g_Na[None,...] * m**3.0 * h * (V - self.E_Na[None,...])
                -self.g_K[None,...] * n**4.0 * (V - self.E_K[None,...])
                -self.g_L[None,...] * (V - self.E_L[None,...]) + Ic[...,None])
        # Coupling term
        dV = torch.sum(self.g_C[None,...]*(V[...,:,None] - V[...,None,:]) / self.C[None,...], dim = -1)
        ydot[...,0::4] += dV

        # m
        ydot[...,1::4] = (self.m_inf - m) / self.tau_m

        # h 
        ydot[...,2::4] = (self.h_inf - h) / self.tau_h

        # n
        ydot[...,3::4] = (self.n_inf - n) / self.tau_n

        J = torch.zeros(y.shape + y.shape[-1:], device = ydot.device)
        
        # V, V
        J[...,0::4,0::4] = torch.diag_embed(1.0 / self.C[None,...] * (
                -self.g_L[None,...] 
                -self.g_Na[None,...]*h*m**3.0
                -self.g_K[None,...]*n**4.0))
        # Coupling term
        J[...,0::4,0::4] -= self.g_C[None,...] / self.C[None,...] 
        
        J[...,0::4,0::4] += torch.eye(self.n_neurons, device = ydot.device).expand(
                *ydot.shape[:-1], -1, -1) * torch.sum(self.g_C / self.C)
        
        # V, m
        J[...,0::4,1::4] = torch.diag_embed(-1.0 / self.C[None,...] * (
                3.0 * self.g_Na[None,...] * h * m**2.0 * (-self.E_Na[None,...] + V)))

        # V, h
        J[...,0::4,2::4] = torch.diag_embed(-1.0 / self.C[None,...] * (
                self.g_Na[None,...] * m**3.0 * (-self.E_Na[None,...] + V)))

        # V, n
        J[...,0::4,3::4] = torch.diag_embed(-1.0 / self.C[None,...] * (
                4.0 * self.g_K[None,...] * n**3.0 * (-self.E_K[None,...] + V)))

        # m, m
        J[...,1::4,1::4] = torch.diag(-1.0 / self.tau_m)

        # h, h 
        J[...,2::4,2::4] = torch.diag(-1.0 / self.tau_h)

        # n, n
        J[...,3::4,3::4] = torch.diag(-1.0 / self.tau_n)
    
        return ydot, J

if __name__ == "__main__":
    # Batch size
    nbatch = 100
    # Time chunk size
    time_block = 50
    # Number of neurons
    neq = 10
    # Number of cycles
    N = 5
    # Number of time steps per cycle
    n = 40
    
    # Setup the driving current
    current, times, discrete_current = driving_current([0.1,1],
            [-1,0.9], 
            [0.5,1.5],
            [0,1],
            [0,1],
            N,
            n,
            n,
            nbatch)

    plt.plot(times.cpu().numpy()[:,::4], discrete_current.cpu().numpy()[:,::4])
    plt.xlabel("Time")
    plt.ylabel("Current")
    plt.show()

    # Setup model and initial conditions
    nsteps = times.shape[0]
    
    model = HodgkinHuxleyCoupledNeurons(
            torch.rand((neq,), device = device),
            torch.rand((neq,), device = device),
            torch.rand((neq,), device = device),
            torch.rand((neq,), device = device),
            torch.rand((neq,), device = device),
            torch.rand((neq,), device = device),
            torch.rand((neq,), device = device),
            torch.rand((neq,), device = device),
            torch.rand((neq,), device = device) * 5.0,
            torch.rand((neq,), device = device),
            torch.rand((neq,), device = device) * 15.0,
            torch.rand((neq,), device = device),
            torch.rand((neq,), device = device) * 10.0,
            torch.rand((neq,), device = device) * 0.01,
            current)

    y0 = torch.rand((nbatch,model.n_equations), device = device)
    

    # Include a backward pass to give a better example of timing
    t1 = time.time()
    res_imp = ode.odeint_adjoint(model, y0, times, block_size = time_block)
    loss = torch.norm(res_imp)
    loss.backward()
    etime = time.time() - t1
    
    print("%i batch size, %i blocked time steps, %i neurons, %i time steps: %f s" % (nbatch, time_block, neq, nsteps, etime))

    plt.plot(times[:,0].detach().cpu().numpy(), res_imp[:,0,0::4].detach().cpu().numpy())
    plt.xlabel("Time")
    plt.ylabel("Voltage")
    plt.show()
