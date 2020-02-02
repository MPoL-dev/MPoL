# MPoL

![](https://github.com/iancze/MPoL/workflows/Python%20package/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/mpol/badge/?version=latest)](https://mpol.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/224543208.svg)](https://zenodo.org/badge/latestdoi/224543208)

A Million Points of Light are needed to synthesize image cubes from interferometers.

Documentation: https://mpol.readthedocs.io/


# Installation

Requires `python > 3.5`. Install the requirements (`numpy`, `scipy`, `torch`, `torchvision`). You may consider [installing PyTorch individually](https://pytorch.org/) if you do/do not require CUDA support and want to fine tune your installation. Or, you could just do 

    pip install -r requirements.txt

after you clone the repository (below).

If you would like to export your images to FITS files, you should also install the `astropy` package.

Then download or `git clone` the MPoL repository and install:

    git clone https://github.com/iancze/MPoL.git
    cd MPoL
    python setup.py install

If you have trouble installing please raise a [github issue](https://github.com/iancze/MPoL/issues).

# Citation 

If you use this package or derivatives of it, please cite

    @software{ian_czekala_2019_3594082,
    author       = {Ian Czekala},
    title        = {iancze/MPoL: Base version},
    month        = dec,
    year         = 2019,
    publisher    = {Zenodo},
    version      = {v0.0.1},
    doi          = {10.5281/zenodo.3594082},
    url          = {https://doi.org/10.5281/zenodo.3594082}
    }


---
Copyright Ian Czekala and contributors 2019