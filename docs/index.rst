.. MPoL documentation master file, created by
   sphinx-quickstart on Sun Dec 22 12:38:55 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Million Points of Light (MPoL)
==============================

MPoL is a Python framework for Regularized Maximum Likelihood (RML) imaging. It is built on top of PyTorch, which provides state of the art auto-differentiation capabilities and optimizers. We focus on supporting spectral line and continuum observations from interferometers like the Atacama Large Millimeter/Submillimeter Array (ALMA) and the Karl G. Jansky Very Large Array (VLA). There is potential to extend the package to work on other Fourier reconstruction problems like sparse aperture masking and kernel phase interferometry.

You can find the source code on `github <https://github.com/iancze/MPoL>`_.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation.rst
   api.rst
   units-and-conventions.rst
   developer-documentation.rst
   
.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/gridder
   usage.rst

.. toctree::
   :hidden:

   changelog.rst

If you use MPoL in your research, please cite us! ::

    @software{MPoL,
    author       = {Ian Czekala, Brianna Zawadzki, and
                    Ryan Loomis},
    title        = {iancze/MPoL: A flexible Python Framework for Regularized Maximum Likelihood Imaging},
    month        = feb,
    year         = 2020,
    publisher    = {Zenodo},
    version      = {v0.0.4},
    doi          = {10.5281/zenodo.3647603},
    url          = {https://doi.org/10.5281/zenodo.3647603}
    }

* :ref:`genindex`
* :ref:`changelog-reference-label`