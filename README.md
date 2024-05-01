# MPoL

[![Tests](https://github.com/MPoL-dev/MPoL/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/MPoL-dev/MPoL/actions/workflows/tests.yml)
[![gh-pages docs](https://github.com/MPoL-dev/MPoL/actions/workflows/gh_docs.yml/badge.svg)](https://mpol-dev.github.io/MPoL/)
[![DOI](https://zenodo.org/badge/224543208.svg)](https://zenodo.org/badge/latestdoi/224543208)



MPoL is a [PyTorch](https://pytorch.org/) *library* built for Regularized Maximum Likelihood (RML) imaging and Bayesian Inference with datasets from interferometers like the Atacama Large Millimeter/Submillimeter Array ([ALMA](https://www.almaobservatory.org/en/home/)) and the Karl G. Jansky Very Large Array ([VLA](https://public.nrao.edu/telescopes/vla/)). 

As a PyTorch *library*, MPoL is designed expecting that the user will write Python code that uses MPoL primitives as building blocks to solve their interferometric imaging workflow, much the same way the artificial intelligence community writes Python code that uses PyTorch layers to implement new neural network architectures (for [example](https://github.com/pytorch/examples)). You will find MPoL easiest to use if you adhere to PyTorch customs and idioms, e.g., feed-forward neural networks, data storage, GPU acceleration, and train/test optimization loops. Therefore, a basic familiarity with PyTorch is considered a prerequisite for MPoL.

MPoL is *not* an imaging application nor a pipeline, though such programs could be built for specialized workflows using MPoL components. We are focused on providing a numerically correct and expressive set of core primitives so the user can leverage the full power of the PyTorch (and Python) ecosystem to solve their research-grade imaging tasks. This is already a significant development and maintenance burden for our small research team, so our immediate scope must necessarily be limited.

* **Documentation** is available at [https://mpol-dev.github.io/MPoL/](https://mpol-dev.github.io/MPoL/)
* **Examples** are available at [https://github.com/MPoL-dev/examples/](https://github.com/MPoL-dev/examples/)

## Citation

If you use this package or derivatives of it, please cite the following two references:

    @software{mpol,
    author       = {Ian Czekala and
                    Jeff Jennings and   
                    Brianna Zawadzki and
                    Ryan Loomis and
                    Kadri Nizam and 
                    Megan Delamer and 
                    Kaylee de Soto and
                    Robert Frazier and
                    Hannah Grzybowski and
                    Mary Ogborn and                    
                    Tyler Quinn},
    title        = {MPoL-dev/MPoL: v0.2.0 Release},
    month        = nov,
    year         = 2023,
    publisher    = {Zenodo},
    version      = {v0.2.0},
    doi          = {10.5281/zenodo.3594081},
    url          = {https://doi.org/10.5281/zenodo.3594081}
    }

and 

    @ARTICLE{2023PASP..135f4503Z,
        author = {{Zawadzki}, Brianna and {Czekala}, Ian and {Loomis}, Ryan A. and {Quinn}, Tyler and {Grzybowski}, Hannah and {Frazier}, Robert C. and {Jennings}, Jeff and {Nizam}, Kadri M. and {Jian}, Yina},
            title = "{Regularized Maximum Likelihood Image Synthesis and Validation for ALMA Continuum Observations of Protoplanetary Disks}",
        journal = {\pasp},
        keywords = {Protoplanetary disks, Submillimeter astronomy, Radio interferometry, Deconvolution, Open source software, 1300, 1647, 1346, 1910, 1866, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
            year = 2023,
            month = jun,
        volume = {135},
        number = {1048},
            eid = {064503},
            pages = {064503},
            doi = {10.1088/1538-3873/acdf84},
    archivePrefix = {arXiv},
        eprint = {2209.11813},
    primaryClass = {astro-ph.EP},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2023PASP..135f4503Z},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

---
Copyright Ian Czekala and contributors 2019-24

A Million Points of Light are needed to synthesize image cubes from interferometers.