pyoptmat: statistical inference for material models
===================================================

pyoptmat is a Python package for using Bayesian inference
to train statistical material constitutive models against
experimental data.
A "material model" is mathematically a parameterized system of ordinary 
differential equations which, integrated through the
experimental conditions as a function of time, returns some simulated
output that can be compared to the test measurements.
pyoptmat uses the `pyro <http://pyro.ai/>`_ package to find
statistical distributions of the model parameters to explain the
variation in the experimental data.

The package is currently geared towards structural material
models that describe how the material reacts to mechanical forces.
However, the underlying mathematical and computational infrastructure
could work for a variety of models for different types of material properties.

You might want to start by installing pyopmat and then look at the
complete tutorial, which takes you through the process of formatting
experimental data, building a model, and training it against the data.
If you are interested in the details of how pyopmat efficiently integrates
ODEs and constructs the sensitivities with an adjoint method, the
ode module documentation outlines that process.  The flowrules, hardening, 
and damage modules contain the current set of prebuilt models, 
again focusing on structural materials.  

After that, the following table provides links to a complete
description of each pyoptmat submodule.

Submodule documentation
-----------------------

.. toctree::

   optimize
   experiments
   ode
   solvers
   models
   flowrules
   hardening
   damage
   temperature
   utility
   bibliography
