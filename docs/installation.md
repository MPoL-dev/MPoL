# Installation and Examples

MPoL requires `python >= 3.10`.

## Using pip

Stable versions are hosted on PyPI. You can install the latest version by

```
$ pip install MPoL
```

Or if you require a specific version of MPoL (e.g., `0.2.0`), you can install via

```
$ pip install MPoL==0.2.0
```

## From source

If you'd like to install the package from source to access the latest development version, download or `git clone` the MPoL repository and install

```
$ git clone https://github.com/MPoL-dev/MPoL.git
$ cd MPoL
$ pip install .
```

If you have trouble installing please raise a [github issue](https://github.com/MPoL-dev/MPoL/issues) with the particulars of your system.

If you're interested in contributing to the MPoL package, please see the [](developer-documentation.md).

## Upgrading

If you installed from PyPI, to upgrade to the latest stable version of MPoL, do

```
$ pip install --upgrade MPoL
```

If you installed from source, update the repository

```
$ cd MPoL
$ git pull
$ pip install .
```

You can determine your current installed version by

```
$ python
>>> import mpol
>>> print(mpol.__version__)
```

## Documentation

The documentation served online ([here](https://mpol-dev.github.io/MPoL/index.html)) corresponds to the `main` branch. This represents the current state of MPoL and is usually the best place to reference MPoL functionality. However, this documentation may be more current than last tagged version or the version you have installed. If you require the new features detailed in the documentation, then we recommend installing the package from source (as above).

In the (foreseeably rare) situation where the latest online documentation significantly diverges from the package version you wish to use (but there are reasons you do not want to build the `main` branch from source), you can access the documentation for that version by [building the older documentation locally](developer-documentation.md#older-documentation-versions)

## Getting Started

As a PyTorch imaging library, there are many things one could do with MPoL. Over at the [MPoL-dev/examples](https://github.com/MPoL-dev/examples/) repository, we've collected example scripts for some of the more common workflows such as diagnostic imaging with {meth}`mpol.gridding.DirtyImager`, imaging with a stochastic gradient descent workflow, and visibility inference with Pyro.
