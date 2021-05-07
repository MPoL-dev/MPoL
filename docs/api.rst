===
API
===

This page documents all of the available components of the MPoL package. If you do not see something that you think should be documented, please raise an `issue <https://github.com/iancze/MPoL/issues>`_.

Utilities
---------

.. automodule:: mpol.utils
    :members:

Coordinates
-----------

.. automodule:: mpol.coordinates
    :members:

Gridding
--------

.. automodule:: mpol.gridding
    :members:

Datasets and Cross-Validation
-----------------------------

.. automodule:: mpol.datasets
    :members:

Images
------

.. automodule:: mpol.images
    :members:

Connectors
----------

.. automodule:: mpol.connectors
    :members:

Losses
------

.. automodule:: mpol.losses
    :members:


Precomposed Modules
--------------------

For convenience, we provide some "precomposed" `modules <https://pytorch.org/docs/stable/notes/modules.html>`_ which may be useful for simple imaging or modeling applications. In general, though, we encourage you to compose your own set of layers if your application requires it. The source code for a precomposed network can provide useful a starting point. We also recommend checking out the PyTorch documentation on `modules <https://pytorch.org/docs/stable/notes/modules.html>`_.

.. automodule:: mpol.precomposed
    :members:
