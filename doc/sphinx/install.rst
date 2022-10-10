Installation
============

pyoptmat is written in pure python and a source package is available on PyPi.
Installing pyoptmat can then be as easy as this:

.. code-block:: console

   $ pip install pyoptmat

The package relies heavily on `pyro <http://pyro.ai/>`_ and 
`torch <https://pytorch.org/>`_ for optimization and inference
algorithms and GPU support.  If you are running on GPUs
then you should follow the platform-specific 
`installation guide for torch <https://pytorch.org/get-started/locally/>`_,
rather than use the CPU-only version that PyPi will provide by default.
