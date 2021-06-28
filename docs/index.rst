.. MPoL documentation master file, created by
   sphinx-quickstart on Sun Dec 22 12:38:55 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Million Points of Light (MPoL)
==============================

|Tests badge|
|Discussions badge|

.. |Tests badge| image:: https://github.com/MPoL-dev/MPoL/actions/workflows/tests.yml/badge.svg
   :target: https://github.com/MPoL-dev/MPoL/actions/workflows/tests.yml

.. |Discussions badge| image:: https://img.shields.io/badge/community-Github%20Discussions-orange
   :target: https://github.com/MPoL-dev/MPoL/discussions

MPoL is a Python framework for Regularized Maximum Likelihood (RML) imaging. It is built on top of PyTorch, which provides state of the art auto-differentiation capabilities and optimizers. We focus on supporting continuum and spectral line observations from interferometers like the Atacama Large Millimeter/Submillimeter Array (ALMA) and the Karl G. Jansky Very Large Array (VLA). There is potential to extend the package to work on other Fourier reconstruction problems like sparse aperture masking and kernel phase interferometry.

To get a sense of how MPoL works, please take a look at the tutorials down below. If you have any questions, please join us on our `Github discussions page <https://github.com/MPoL-dev/MPoL/discussions>`__.

If you'd like to help build the MPoL package, please check out the :ref:`developer-documentation-label` to get started. For more information about the constellation of packages supporting RML imaging and modeling, check out the MPoL-dev organization `website <https://mpol-dev.github.io/>`_ and `github <https://github.com/MPoL-dev>`__ repository hosting the source code.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation.rst
   units-and-conventions.rst
   developer-documentation.rst
   api.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/PyTorch
   tutorials/gridder
   tutorials/optimization
   tutorials/crossvalidation
   tutorials/gpu_setup.rst
   tutorials/initializedirtyimage
   tutorials/HD143006_Part_2

.. toctree::
   :hidden:

   changelog.rst

If you use MPoL in your research, please cite us! ::

   @software{ian_czekala_2021_4939048,
   author       = {Ian Czekala and
                  Brianna Zawadzki and
                  Ryan Loomis and
                  Hannah Grzybowski and
                  Robert Frazier and
                  Tyler Quinn},
   title        = {MPoL-dev/MPoL: v0.1.1 Release},
   month        = jun,
   year         = 2021,
   publisher    = {Zenodo},
   version      = {v0.1.1},
   doi          = {10.5281/zenodo.4939048},
   url          = {https://doi.org/10.5281/zenodo.4939048}
   }

* :ref:`genindex`
* :ref:`changelog-reference-label`
