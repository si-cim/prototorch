.. ProtoTorch API Reference

ProtoTorch API Reference
======================================

Datasets
--------------------------------------

Common Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: prototorch.datasets
   :members:


Abstract Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Abstract Datasets are used to build your own datasets.

.. autoclass:: prototorch.datasets.abstract.NumpyDataset
   :members:

Functions
--------------------------------------

**Dimensions:**

- :math:`B` ... Batch size
- :math:`P` ... Number of prototypes
- :math:`n_x` ... Data dimension for vectorial data
- :math:`n_w` ... Data dimension for vectorial prototypes

Activations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: prototorch.functions.activations
   :members:
   :exclude-members: register_activation, get_activation
   :undoc-members:

Distances
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: prototorch.functions.distances
   :members:
   :exclude-members: sed
   :undoc-members:

Modules
--------------------------------------
.. automodule:: prototorch.modules
   :members:
   :undoc-members:

Utilities
--------------------------------------
.. automodule:: prototorch.utils
   :members:
   :undoc-members:
