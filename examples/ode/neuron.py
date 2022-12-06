#!/usr/bin/env python

import torch
import torch.nn as nn

import torch.jit
from torch.profiler import ProfilerActivity

import matplotlib.pyplot as plt

from pyoptmat import ode, experiments, utility

torch.set_default_tensor_type(torch.DoubleTensor)

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def driving_current(I_max, R, rate, thold, chold, Ncycles, nload,
        nhold, neq):
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
    
    return utility.BatchTimeSeriesInterpolator(
            times.to(device), I.to(device)), times.to(device)

class HodgkinHuxleyCoupledNeurons(torch.nn.Module):
    """
        From Schwemmer and Lewis 2012

        dX / dt = 
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
        Ic = self.I(t)

        V = y[...,0::4]
        m = y[...,1::4]
        h = y[...,2::4]
        n = y[...,3::4]

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
        J[...,0::4,0::4] += torch.repeat_interleave(torch.eye(self.n_neurons, 
            device = ydot.device).unsqueeze(0), ydot.shape[0], 0
                ) * torch.sum(self.g_C / self.C)
        
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
    nbatch = 100
    neq = 10
    N = 5
    n = 10
    
    current, times = driving_current([0.1,1],
            [-1,0.9], 
            [0.5,1.5],
            [0,1],
            [0,1],
            N,
            n,
            n,
            nbatch)

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
    
    with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/tracer'),
            with_stack=True):
    
        for i in range(1):
            res_imp = ode.odeint_adjoint(model, y0, times)
            loss = torch.norm(res_imp)
            loss.backward()

    #plt.plot(times[:,0].detach().numpy(), res_imp[:,0,0::4].detach().numpy())
    #plt.show()
