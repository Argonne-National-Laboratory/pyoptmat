Other sample problems
=====================

Optimizing deterministic material models
----------------------------------------

The example in `examples/structural-inference/tension/deterministic`
repeats the example described in the :doc:`detailed tutorial <structural_tutorial>` except targeting a
deterministic, instead of a statistical, model.
The model uses the same synthetic data used in the statistical inference
tutorial but fits a deterministic best fit model to the data.  The example
illustrates how to use the package to train deterministic models and also
demonstrates the potential utility of the L-BFGS optimizer when (a) you are
training deterministic models and (b) you have sufficient GPU memory 
available.

.. literalinclude:: /../../examples/structural-inference/tension/maker.py

The example in `examples/structural-inference/creep-fatigue/deterministic`
repeats a similar example using synthetic cyclic (creep-fatigue) test
data and a :py:class:`pyoptmat.hardening.ChabocheHardeningModel` viscoplastic material model.

Tutorial example for cyclic load
--------------------------------

`examples/structural-inference/creep-fatigue/statistical` basically
repeats the :doc:`tutorial <structural_tutorial>`, but using
synthetic cyclic test data a :py:class:`pyoptmat.hardening.ChabocheHardeningModel` viscoplastic model capable
of capturing kinematic hardening behavior.

Neural ODE example
------------------

Inferring the response of a coupled mass-spring-damper system using
trajectory data (1) a physical ODE model and (2) a neural ODE model.

.. literalinclude:: /../../examples/ode/damping.py

Implicit versus explicit integration
------------------------------------

The example in `examples/ode/vanderpohl.py` demonstrates the need for
an implicit integration scheme when integrating stiff systems of ODEs,
including structural constitutive models.  The example
uses the :doc:`ode <ode>` package to integrate the classic van der Pol
equation with different parameters to demonstrate the need for implicit
time integration as the equations become stiff.  The results of the
calculations are differentiable using either direct torch AD or the adjoint
methods.

.. literalinclude:: /../../examples/ode/vanderpohl.py

Statistical inference with a simple ODE
---------------------------------------

This example (`examples/ode/trajectory.py`) demonstrates the basic
application of pyoptmat to find parameter distributions for systems of
stochastic ODEs that match uncertain data.  This example demonstrates
these concepts using trajectories from a canon fired at a random angle
and speed, drawn from known distributions.  The example tries to infer the
distributions of angle and speed given some number of observed trajectories.
This problem is small enough that you can run it using either the adjoint
method or AD to calculate the sensitivities, to compare the relative
performance of both approaches.

.. literalinclude:: /../../examples/ode/trajectory.py

Hogdkin-Huxley coupled neuron model
-----------------------------------

An example, scalable system of ODEs for benchmarking.

.. literalinclude:: /../../examples/ode/neuron.py

Temperature dependent parameters
--------------------------------

`examples/structural-material-models/temperature_dependence.py` illustrates
how to setup and run structural material models with temperature dependent
parameters.

.. literalinclude:: /../../examples/structural-material-models/temperature_dependence.py

Stress and strain control
-------------------------

The example in `examples/structural-material-models/stress_strain_control.py`
demonstrates how to run simulations of structural tests conducted under
strain and stress control.  pyoptmat can run both types of tests and 
correctly provide derivatives using the adjoint method

.. literalinclude:: /../../examples/structural-material-models/stress_strain_control.py

Creep tests
-----------

`examples/structural-material-models/stress_control_creep.py` is an example
of simulating stress-controlled creep tests, with the idea of using such
data to train constitutive models.

.. literalinclude:: /../../examples/structural-material-models/stress_control_creep.py

AD versus adjoint comparison
----------------------------

`examples/performance/ad_vs_adjoint.py` compares the efficiency of computing
the parameter gradients using the AD versus the adjoint method.  
It demonstrates that the adjoint method is always better for practical problems:
it is both faster and requires far less memory on the device.

.. literalinclude:: /../../examples/performance/ad_vs_adjoint.py
