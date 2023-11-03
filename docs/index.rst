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

.. raw:: html

   <!-- Place this tag where you want the button to render. -->
   <a class="github-button" href="https://github.com/MPoL-dev/MPoL" data-color-scheme="no-preference: light; light: light; dark: dark_dimmed;" data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star MPoL-dev/MPoL on GitHub">Star</a>
   <a class="github-button" href="https://github.com/MPoL-dev/MPoL/discussions" data-color-scheme="no-preference: light; light: light; dark: dark_dimmed;" data-icon="octicon-comment-discussion" data-size="large" aria-label="Discuss MPoL-dev/MPoL on GitHub">Discuss</a>


MPoL is a Python framework for Regularized Maximum Likelihood (RML) imaging. It is built on top of PyTorch, which provides state of the art auto-differentiation capabilities and optimizers. We focus on supporting continuum and spectral line observations from interferometers like the Atacama Large Millimeter/Submillimeter Array (ALMA) and the Karl G. Jansky Very Large Array (VLA). There is potential to extend the package to work on other Fourier reconstruction problems like sparse aperture masking and other forms of optical interferometry.

To get a sense of how MPoL works, please take a look at the :ref:`rml-intro-label` and then the tutorials down below. If you have any questions, please join us on our `Github discussions page <https://github.com/MPoL-dev/MPoL/discussions>`__.

If you'd like to help build the MPoL package, please check out the :ref:`developer-documentation-label` to get started. For more information about the constellation of packages supporting RML imaging and modeling, check out the MPoL-dev organization `website <https://mpol-dev.github.io/>`_ and `github <https://github.com/MPoL-dev>`__ repository hosting the source code.



.. toctree::
   :maxdepth: 2
   :caption: User Guide

   rml_intro.md
   installation.md
   units-and-conventions.md
   developer-documentation.md
   api.rst

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   ci-tutorials/PyTorch
   ci-tutorials/gridder
   ci-tutorials/optimization
   ci-tutorials/loose-visibilities
   ci-tutorials/crossvalidation
   ci-tutorials/gpu_setup.rst
   ci-tutorials/initializedirtyimage
   large-tutorials/HD143006_part_1
   large-tutorials/HD143006_part_2
   ci-tutorials/fakedata
   large-tutorials/pyro

.. toctree::
   :hidden:

   changelog.md

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
