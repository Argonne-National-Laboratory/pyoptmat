pyoptmat: statistical inference for material models
===================================================

pyoptmat is a Python package for using Bayesian inference
to train statistical material constitutive models against
experimental data.
A "material model" is mathematically a parameterized system of ordinary 
differential equations which, integrated through the
experimental conditions, returns some simulated
output that can be compared to the test measurements.
pyoptmat uses the `pyro <http://pyro.ai/>`_ package to find
statistical distributions of the model parameters to explain the
variation in the experimental data.

As an example, consider a collection of tension test data on several samples
of a material.  The test measurements have some variation caused by
manufacturing variability and uncertainty in the experimental controls
and measurements.

pyoptmat aims to make training a statistical model to capture these
variations easy.  The image shows the results of training a simple
material model to the test data.  The trained statistical model captures
the variability in the experimental data and can then be used to
translate this uncertainty to models of engineering components.  
Transferring uncertainty quantified in experimental measurements
to predictions of uncertainty in engineering applications is the main 
reason pyoptmat was developed.

.. figure:: figures/demonstration.png
   :width: 400
   :alt: Example of fitting a statistical model to data

   Example of what pyoptmat can do: take uncertain experimental data (dashed lines)
   and use it to train a statistical material model that can predict and extrapolate
   that uncertaintly (solid line and prediction interval).

The package is currently geared towards structural material
models that describe how materials react to mechanical forces.
Mathematically then, these models are systems of stochastic ordinary
differential equations.
However, the underlying mathematical and computational infrastructure
could work for a variety of models for different types of material properties.


Getting started
---------------

The :doc:`highlights <highlights>` section contains a high-level
overview of pyoptmat can do.

To get started start by :doc:`installing pyopmat <install>` and then look at the
:doc:`complete tutorial <structural_tutorial>`, which takes you through the process of formatting
experimental data, building a model, and training it against the data.
If you are interested in the details of how pyopmat efficiently integrates
ODEs and constructs the sensitivities with an adjoint method, the
:doc:`ode module <ode>` documentation outlines that process.  
The :doc:`flowrules <flowrules>`, :doc:`hardening <hardening>`, 
and :doc:`damage <damage>` modules contain the current set of prebuilt models, 
focusing on structural materials.  

.. toctree::
   :maxdepth: 1
   
   highlights
   install
   structural_tutorial
   examples
   bibliography

Submodule documentation
-----------------------

After that, the following table provides links to a complete
API description of each pyoptmat submodule.

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

