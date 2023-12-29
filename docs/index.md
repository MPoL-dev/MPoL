# Million Points of Light (MPoL)

[![Tests](https://github.com/MPoL-dev/MPoL/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/MPoL-dev/MPoL/actions/workflows/tests.yml)
[![gh-pages docs](https://img.shields.io/badge/community-Github%20Discussions-orange)](https://github.com/MPoL-dev/MPoL/discussions)

MPoL is a [PyTorch](https://pytorch.org/) *library* built for Regularized Maximum Likelihood (RML) imaging and Bayesian Inference with datasets from interferometers like the Atacama Large Millimeter/Submillimeter Array ([ALMA](https://www.almaobservatory.org/en/home/)) and the Karl G. Jansky Very Large Array ([VLA](https://public.nrao.edu/telescopes/vla/)). 

As a PyTorch *library*, MPoL expects that the user will write Python code to link MPoL primitives as building blocks to solve their interferometric imaging workflow, much the same way the artificial intelligence community uses PyTorch layers to build new neural network architectures (for [example](https://github.com/pytorch/examples)). You will find MPoL easiest to use if you emulate PyTorch customs and idioms, e.g., feed-forward neural networks, data storage, GPU acceleration, and train/test optimization loops. Therefore, a basic familiarity with PyTorch is considered a prerequisite for MPoL.

MPoL is *not* an imaging application nor a pipeline, though MPoL components could be used to build specialized workflows. We are focused on providing a numerically correct and expressive set of core primitives so the user can leverage the full power of the PyTorch (and Python) ecosystem to solve their research-grade imaging tasks. This is already a significant development and maintenance burden for the limited resources of our small research team, so our immediate scope must necessarily be limited.

To get a sense of what background material MPoL assumes, please look at the [](background.md). If the package is right for your needs, follow the [installation instructions](installation.md).

This documentation covers the API and a short set of tutorials demonstrating key components of the MPoL library. Longer examples demonstrating how one might use MPoL components to build an imaging workflow are packaged together in the [MPoL-dev/examples](https://github.com/MPoL-dev/examples) repository.

If you'd like to help build the MPoL package, please check out the [](developer-documentation.md) to get started. For more information about the constellation of packages supporting RML imaging and modeling, check out the MPoL-dev organization [website](https://mpol-dev.github.io/) and [github](https://github.com/MPoL-dev) repository hosting the source code. If you have any questions, please ask us on our [Github discussions page](https://github.com/MPoL-dev/MPoL/discussions). 

*If you use MPoL in your research, please cite us!* See <https://github.com/MPoL-dev/MPoL#citation> for the citation.

```{toctree}
:caption: User Guide
:maxdepth: 2

background
installation
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
```

```{toctree}
:caption: API
:maxdepth: 2

api/coordinates
api/datasets
api/fourier
api/gridding
api/images
api/losses
api/geometry
api/utilities
api/precomposed
api/train_test
api/plotting
api/crossval
api/analysis
```

```{toctree}
:caption: Reference
:maxdepth: 2

units-and-conventions.md
developer-documentation.md
changelog.md
```

- {ref}`genindex`
- {ref}`changelog-reference-label`
