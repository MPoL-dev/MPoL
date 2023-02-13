.. _api-reference-label:

===
API
===

This page documents all of the available components of the MPoL package. If you do not see something that you think should be documented, please raise an `issue <https://github.com/iancze/MPoL/issues>`_.

Utilities
---------

.. automodule:: mpol.utils


Coordinates
-----------

.. automodule:: mpol.coordinates

Geometry
--------

.. automodule:: mpol.geometry

Gridding
--------

.. automodule:: mpol.gridding

Datasets
--------

.. automodule:: mpol.datasets

Images
------

.. automodule:: mpol.images

Fourier
-------

.. automodule:: mpol.fourier


Precomposed Modules
--------------------

For convenience, we provide some "precomposed" `modules <https://pytorch.org/docs/stable/notes/modules.html>`_ which may be useful for simple imaging or modeling applications. In general, though, we encourage you to compose your own set of layers if your application requires it. The source code for a precomposed network can provide useful a starting point. We also recommend checking out the PyTorch documentation on `modules <https://pytorch.org/docs/stable/notes/modules.html>`_.

.. automodule:: mpol.precomposed

Losses
------

.. automodule:: mpol.losses


Training and testing
--------------------

.. automodule:: mpol.training


Connectors
----------

The objects in the Images and Precomposed modules are focused on bringing some image-plane model to the space of the data, where the similarity of the model visibilities to the data visibilities will be evaluated by a negative log-likelihood loss. In some situations, though, it is useful to have access to the residual visibilities directly. For example, for visualization or debugging purposes.

Connectors are a PyTorch layer to help compute those residual visibilities (on a gridded form).

.. automodule:: mpol.connectors


Cross-validation
----------------

.. automodule:: mpol.crossval


Plotting
--------

.. automodule:: mpol.plot
