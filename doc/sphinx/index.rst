pyoptmat: statistical inference for material models
===================================================

*pyoptmat takes uncertain experimental data and uses it to train 
a statistical material model that can predict and extrapolate that
variability to new loading conditions.*

pyoptmat is a Python package for using Bayesian inference
to train statistical material constitutive models against
experimental data.  Material models describe how
the material responds to some external stimulus.  For example,
structural material models describe how the material reacts to
mechanical forces.  Mathematically, these models are systems of ordinary
differential equations.  pyoptmat trains stoachastic ordinary differential
equations to match the variability in the experimental data.

As an example, consider a collection of tension test data from 
several samples of a material.  Test measurements vary due to 
manufacturing variability and uncertainty in the experimental controls
and measurements.  pyoptmat uses the `pyro <http://pyro.ai/>`_ package to find
statistical distributions of the model parameters to explain the
variation in the experimental data.

The image below shows pyoptmat makes the process of a training a statistical model simple.
The trained statistical model captures
the variability in the experimental data and then can
translate this uncertainty to models of engineering components.  

.. figure:: figures/demonstration.png
   :width: 400
   :alt: Example of fitting a statistical model to data

   pyoptmat takes uncertain experimental data and uses it to train a statistical material 
   model that can predict and extrapolate that uncertainty.

pyoptmat features
-----------------

- A complete open source solution for training statistical, ODE models against 
  uncertain data.
- Efficient backward pass/gradient calculation using the adjoint method.  This
  approach vastly outperforms automatic differentiation for time series data. 
- Blocked time integration for both the forward and backward/adjoint passes, which 
  vectorizes/parallelizes integrating ODEs through time.
- Implicit time integration algorithms suitable for material models represents
  as stiff systems of ordinary differential equations.
- Prebuilt model components aimed at high temperature structural materials.
- An abstraction that makes it easy to train both deterministic and statistical
  models, starting from the same base material model form.
- Examples and tests to help you get started.

Getting started
---------------

To get started, simply :doc:`install pyopmat <install>` and then review the
:doc:`complete tutorial <structural_tutorial>`, which takes you through the process of formatting
experimental data, building a model, and training it against the data.
If you are interested in the details of how pyopmat efficiently integrates
ODEs and constructs the sensitivities using an adjoint method, the
:doc:`ode module <ode>` documentation outlines that process.  
The :doc:`flowrules <flowrules>`, :doc:`hardening <hardening>`, 
and :doc:`damage <damage>` modules are where to find the current set of prebuilt models, 
focusing on structural materials.  

.. toctree::
   :maxdepth: 1
   
   install
   structural_tutorial
   examples
   bibliography

Submodule documentation
-----------------------

The following are links to the complete API descriptions of each pyoptmat submodule.

.. toctree::

   optimize
   experiments
   ode
   chunktime
   solvers
   models
   flowrules
   hardening
   damage
   temperature
   utility

