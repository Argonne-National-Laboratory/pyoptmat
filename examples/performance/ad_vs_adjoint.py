#!/usr/bin/env python3

"""
    A comparison of the efficiency of Pytorch AD versus the adjoint method
    for calculating the parameter gradient of a model.

    The demo runs `n_tests` tension tests using `n_steps` timesteps to
    integrate each test.  Given these two size parameter as input, the 
    code:
    
    1. Generates the input tensors describing the time/strain history of
       each tensile test
    2. Runs the same process twice, once using Pytorch AD to calculate the
       gradient and then again using the adjoint method

       a. Integrate the tests through time (forward pass)
       b. Calculate the norm of the resulting integrated stresses, which 
          is a reasonable surrogate for the actual loss functions used
          during training
       c. Calculate the gradient of that norm value with respect to the model
          parameters

    3. Compare the walltime and memory costs of running the analysis using
       the two methods.

    Running this script on a NVIDIA GeForce RTX 3070 Ti gives the following
    results for `n_tests = 500`:
    
    ======= ============= ============= ================== =================
    n_time  AD, time (s)  AD, mem (MB)  Adjoint, time (s)  Adjoint, mem (MB)
    ======= ============= ============= ================== =================
    100     9.266         274.4         7.057              18.42
    200     16.93         475.6         12.95              37.58
    300     24.43         674.8         18.92              55.21
    400     31.96         872.0         25.01              73.45    
    500     40.59         1062          31.97              93.76
    750     60.26         1527          46.13              137.8
    1000    77.55         2027          62.43              188.0
    ======= ============= ============= ================== =================

    Running for fixed `n_time = 100` gives:

    ======== ============= ============= ================== =================
    n_tests  AD, time (s)  AD, mem (MB)  Adjoint, time (s)  Adjoint, mem (MB)
    ======== ============= ============= ================== =================
    100      8.963         68.27         6.859              3.781
    200      9.193         131.9         6.993              7.512
    300      9.084         171.7         6.962              11.10
    400      9.257         234.6         6.947              14.83 
    500      9.175         274.4         7.004              18.42
    750      9.280         410.1         7.034              27.60
    1000     9.424         545.2         7.084              37.72
    5000     9.984         2686.7        7.613              187.3
    ======== ============= ============= ================== =================
"""

# Size of problem to run
n_tests = 500
n_time = 500

import sys

sys.path.append("../..")

import time

from pyoptmat import models, flowrules, temperature, experiments, hardening, optimize
from pyoptmat.temperature import ConstantParameter as CP

import torch

torch.set_default_tensor_type(torch.DoubleTensor)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def model_maker(E, R, d, n, eta, s0, C, g, **kwargs):
    """
    Make the model
    """
    iso = hardening.VoceIsotropicHardeningModel(CP(R), CP(d))
    kin = hardening.ChabocheHardeningModel(CP(C), CP(g))
    flow = flowrules.IsoKinViscoplasticity(CP(n), CP(eta), CP(s0), iso, kin)
    mod = models.InelasticModel(CP(E), flow)
    return models.ModelIntegrator(mod, **kwargs).to(device)


if __name__ == "__main__":
    # Model parameters to use
    E = torch.tensor(150000.0, device=device)
    R = torch.tensor(200.0, device=device)
    d = torch.tensor(5.0, device=device)
    n = torch.tensor(7.0, device=device)
    eta = torch.tensor(300.0, device=device)
    s0 = torch.tensor(50.0, device=device)
    C = torch.tensor([10000.0, 5000.0], device=device)
    g = torch.tensor([100.0, 50.0], device=device)
    names = ["E", "R", "d", "n", "eta", "s0", "C", "g"]
    values = [E, R, d, n, eta, s0, C, g]

    # Setup the input data
    rates = torch.ones(n_tests, device=device) * 1.0e-5
    temps = torch.zeros(n_tests, device=device)
    elimits = torch.ones(n_tests, device=device) * 0.2
    times, strains, temps, cycles = experiments.make_tension_tests(
        rates, temps, elimits, n_time
    )

    data = torch.stack((times, temps, strains)).to(device)
    types = torch.tensor([experiments.exp_map["tensile"]] * n_tests, device=device)
    control = torch.tensor([experiments.control_map["strain"]] * n_tests, device=device)

    print("Running example with size %i x %i" % (n_tests, n_time))
    print("")

    # Do this twice (once for AD, once for adjoint)
    for use_adjoint in [False, True]:
        # Get the actual model
        maker = lambda *p: model_maker(*p, use_adjoint=use_adjoint)
        model = optimize.DeterministicModel(maker, names, values)

        # Start timing/monitoring
        tstart = time.time()
        torch.cuda.reset_peak_memory_stats(device)

        # Run the forward integration
        pred = model(data, cycles, types, control)

        # Calculate the loss surrogate
        loss = torch.norm(pred)

        # Calculate the gradient
        loss.backward()

        # Stop monitoring
        tend = time.time()
        mem = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Print statistics
        if use_adjoint:
            print("Adjoint method:")
        else:
            print("Torch AD:")
        print("\tWalltime (s): %f" % (tend - tstart))
        print("\tMax memory use (MB): %f" % mem)
        print("")
