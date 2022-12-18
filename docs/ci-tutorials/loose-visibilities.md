---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell}
:tags: [hide-cell]
%matplotlib inline
%run notebook_setup
```

# Using the NUFFT to predict individual visibilities

Typical interferometric datasets from ALMA may contain over 100,000 individual visibility measurements. As you saw in the [MPoL optimization introduction](optimization.md), the basic MPoL workflow is to perform a weighted average of these individual visibilities to the $u,v$ Fourier grid defined by a {class}`mpol.coordinates.GridCoords` object and calculate equivalent weights for those grid cells. This means that any forward-modeling application of MPoL need only carry the image model to the gridded Fourier plane, since likelihood evaluations can be done on a cell-to-cell basis. Depending on the dimensions of the Fourier grid, this can dramatically reduce the effective size of the dataset and speed up the computational time.

For some applications, though, it may be desireable to keep the dataset as individual visibility points and instead carry the image model all the way to these discrete points. This allows a direct comparison in the space of the ungridded data. Moreover, this allows a more thorough examination of visibility residuals for at least two reasons:

1. Residual visibilities may be "regridded" using some form of Briggs or natural weighting and then imaged with a dirty image. This improves point source sensitivity and may make faint residual structures easier to detect.
2. Individual visibility data might have relevant metadata that would otherwise be destroyed in a cell-averaging process. For example, perhaps visibilities from a particular execution block were incorrectly calibrated. It would be possible to spot such a data-defect if residual scatter were examined on a per-visibility, per-execution block basis.

In this tutorial, we will explore the {class}`mpol.fourier.NuFFT` object and how it may be used to compute individual visibilities.

## Mock image  

We will use $u,v$ locations from the same mock-dataset as before. So we'll start by importing the relevant functions

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.utils.data import download_file
```

and the relevant MPoL modules

```{code-cell}
from mpol import coordinates, gridding, losses, precomposed, utils
```

and loading the dataset

```{code-cell}
# load the mock dataset of the ALMA logo
fname = download_file(
    "https://zenodo.org/record/4930016/files/logo_cube.noise.npz",
    cache=True,
    show_progress=True,
    pkgname="mpol",
)

# this is a multi-channel dataset... 
d = np.load(fname)
uu = d["uu"]
vv = d["vv"]
weight = d["weight"]
data = d["data"]
data_re = np.real(data)
data_im = np.imag(data)
```

```{code-cell}
# define the image dimensions, as in the previous tutorial
coords = coordinates.GridCoords(cell_size=0.005, npix=800)
```

This dataset has multiple channels to it, which we'll use to demonstrate some of the various features of the {class}`mpol.fourier.NuFFT` object.

## The {class}`mpol.fourier.NuFFT` object.

The {class}`mpol.fourier.NuFFT` object relies upon the functionality provided by the [TorchKbNuFFT package](https://torchkbnufft.readthedocs.io/en/stable/). Before going further, we encourage you to read the API documentation of the {class}`mpol.fourier.NuFFT` object itself. There are two main modes of functionality to consider for this object, which depend on the dimensionality of your baseline arrays.

Paraphrasing from the {class}`mpol.fourier.NuFFT` API documentation, 

* If you provide baseline arrays ``uu`` and ``vv`` with a shape of (``nvis``), then it will be assumed that the spatial frequencies can be treated as constant with channel. This is likely a safe assumption for most spectral line datasets (but one you can check yourself using {func}`mpol.fourier.safe_baseline_constant_meters` or {func}`mpol.fourier.safe_baseline_constant_kilolambda`). 
* If the ``uu`` and ``vv`` have a shape of (``nchan, nvis``), then it will be assumed that the spatial frequencies are different for each channel, and the spatial frequencies provided for each channel will be used.





## Gridder

Send the visibilities to the normal gridder object

```{code-cell}
gridder = gridding.Gridder(
    coords=coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data_re,
    data_im=data_im,
)
```

For both applications, 

This tutorial explains how to use the non-uniform FFT to take an image plane model to the loose visibilities of the dataset.

Different modes (parallelizing over channel with coil or with batch dimensionality).


# timing tests -- possible with actual dataset
