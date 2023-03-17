#!/usr/bin/env python3

import sys
sys.path.append('../../..')

import xarray as xr
import torch
import pyro
import matplotlib.pyplot as plt

import tqdm

from pyoptmat import optimize, experiments, models, flowrules, hardening, temperature, scaling

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

# Elastic constants, don't infer
E_poly = torch.tensor([-5.76107011e-05,  7.52478382e-02, -9.98116448e+01,  2.19193150e+05], device = device)
E = temperature.PolynomialScaling(E_poly)

if __name__ == "__main__":
    # Time chunking
    time_chunk_size = 5

    # Load in the data
    input_data = xr.open_dataset("data.nc")
    data, results, cycles, types, control = experiments.load_results(
            input_data)

    # Figure out our temperature control points
    ncontrol = 5
    Tcontrol = torch.linspace(torch.min(data[1]), torch.max(data[1]), 
            ncontrol, device = device)
    
    # Model making function
    def make(n_vals, eta_vals, theta_vals, tau_vals, scale_functions = [lambda x: x] * 4, **kwargs):
        n = temperature.PiecewiseScaling(Tcontrol, n_vals, values_scale_fn = scale_functions[0])
        eta = temperature.PiecewiseScaling(Tcontrol, eta_vals, values_scale_fn = scale_functions[1])
        theta = temperature.PiecewiseScaling(Tcontrol, theta_vals, values_scale_fn = scale_functions[2])
        tau = temperature.PiecewiseScaling(Tcontrol, tau_vals, values_scale_fn=scale_functions[3])

        isotropic = hardening.Theta0VoceIsotropicHardeningModel(
                tau,
                theta
                )

        kinematic = hardening.NoKinematicHardeningModel()
        flowrule = flowrules.IsoKinViscoplasticity(
                n, 
                eta, 
                temperature.ConstantParameter(torch.tensor(0.0, device = device)),
                isotropic,
                kinematic
        )
        model = models.InelasticModel(
                E,
                flowrule
        )

        return models.ModelIntegrator(model, **kwargs).to(device)
    
    # Setup initial guesses
    names = ["n_vals", "eta_vals", "theta_vals", "tau_vals"]
    scale_functions = [
            scaling.BoundedScalingFunction(torch.tensor(1.1, device = device), torch.tensor(15.0, device = device)),
            scaling.BoundedScalingFunction(torch.tensor(20.0, device = device), torch.tensor(1000.0, device = device)),
            scaling.BoundedScalingFunction(torch.tensor(200.0, device = device), torch.tensor(20000.0, device = device)),
            scaling.BoundedScalingFunction(torch.tensor(10.0, device = device), torch.tensor(1000.0, device = device))
            ]
    loc_loc_priors = [
            scale_functions[0].unscale(torch.ones(ncontrol, device = device) * 5.0),
            scale_functions[1].unscale(torch.ones(ncontrol, device = device) * 500.0),
            scale_functions[2].unscale(torch.ones(ncontrol, device = device) * 3000.0),
            scale_functions[3].unscale(torch.ones(ncontrol, device = device) * 500.0)
            ]
    loc_scale_priors = [0.1 * l for l in loc_loc_priors]
    scale_scale_priors = [0.1 * l for l in loc_loc_priors]
    eps_prior = torch.tensor(5.0, device = device)

    print("Prior values")
    for n,l,s1, s2 in zip(names, loc_loc_priors, loc_scale_priors, scale_scale_priors):
        print(n)
        print("\tloc: %s" % str(l.cpu()))
        print("\tloc scale: %s" % str(s1.cpu()))
        print("\tscale scale: %s" % str(s2.cpu()))

    # Do a quick check on the reasonableness of the priors
    test = optimize.DeterministicModel(lambda *args, **kwargs: make(*args, 
        scale_functions=scale_functions, block_size = time_chunk_size, **kwargs),
        names, loc_loc_priors)
    
    with torch.no_grad():
        test_results = test(data.to(device), cycles.to(device), types.to(device),
                control.to(device))
    plt.plot(data[-1].cpu().numpy(), test_results.cpu().numpy())
    plt.show()

    # Create the actual model
    model = optimize.HierarchicalStatisticalModel(
            lambda *args, **kwargs: make(*args, scale_functions = scale_functions,
                block_size = time_chunk_size, **kwargs), 
            names, loc_loc_priors, loc_scale_priors, scale_scale_priors, eps_prior
    ).to(device)

    # Get the guide
    guide = model.make_guide()

    # 5) Setup the optimizer and loss
    lr = 1.0e-3
    g = 1.0
    niter = 500
    num_samples = 1

    optimizer = pyro.optim.ClippedAdam({"lr": lr})

    ls = pyro.infer.Trace_ELBO(num_particles=num_samples)

    svi = pyro.infer.SVI(model, guide, optimizer, loss=ls)

    # Actually infer
    t = tqdm.tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_hist = []
    for i in t:
        loss = svi.step(data.to(device), cycles.to(device), types.to(device),
                control.to(device), results.to(device))
        loss_hist.append(loss)
        t.set_description("Loss %3.2e" % loss)
