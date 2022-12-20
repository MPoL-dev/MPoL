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

# Likelihood functions and model visibilities

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
from mpol import coordinates, gridding, losses, precomposed, utils, images, fourier
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
nchan = len(uu)
```

```{code-cell}
# define the image dimensions, as in the previous tutorial
coords = coordinates.GridCoords(cell_size=0.005, npix=800)
```

This dataset has multiple channels to it, which we'll use to demonstrate some of the various features of the {class}`mpol.fourier.NuFFT` object.

## The {class}`mpol.fourier.NuFFT` object

The {class}`mpol.fourier.NuFFT` object relies upon the functionality provided by the [TorchKbNuFFT package](https://torchkbnufft.readthedocs.io/en/stable/). Before going further, we encourage you to read the API documentation of the {class}`mpol.fourier.NuFFT` object itself. There are two main modes of functionality to consider for this object, which depend on the dimensionality of your baseline arrays.

Paraphrasing from the {class}`mpol.fourier.NuFFT` API documentation,

* If you provide baseline arrays ``uu`` and ``vv`` with a shape of (``nvis``), then it will be assumed that the spatial frequencies can be treated as constant with channel. This is likely a safe assumption for most spectral line datasets (but one you can check yourself using {func}`mpol.fourier.safe_baseline_constant_meters` or {func}`mpol.fourier.safe_baseline_constant_kilolambda`).
* If the ``uu`` and ``vv`` have a shape of (``nchan, nvis``), then it will be assumed that the spatial frequencies are different for each channel, and the spatial frequencies provided for each channel will be used.

Let's use the {func}`mpol.fourier.safe_baseline_constant_kilolambda` routine to check the status of the arrays in this dataset.

```{code-cell}
fourier.safe_baseline_constant_kilolambda(uu, vv, coords, uv_cell_frac=0.05)
```

So, we would be safe to proceed with using the $u,v$ values from a single channel as representative. Let's proceed with this assumption

```{code-cell}
chan = 4
uu_chan = uu[chan]
vv_chan = vv[chan]
```

and then use these values to initialize a {class}`mpol.fourier.NuFFT` object

```{code-cell}
nufft = fourier.NuFFT(coords=coords, nchan=nchan, uu=uu_chan, vv=vv_chan)
```

Now let's put the NuFFT aside for a moment while we initialize the {class}`mpol.gridding.Gridder` object and create an image for use in the forward model.

## Compared to the Gridder object

As before, we simply send the visibilities to the object and export a {class}`mpol.datasets.GriddedDataset`

```{code-cell}
gridder = gridding.Gridder(
    coords=coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data_re,
    data_im=data_im,
)

gridded_dset = gridder.to_pytorch_dataset()
```

And we can initialize a :class:`mpol.fourier.FourierCube`

```{code-cell}
flayer = fourier.FourierCube(coords=coords)
```

## Image-plane forward model

RML is fundamentally a forward-modeling application. To test and compare both the NuFFT and gridded approaches, we'll use the same forward model.

We could start from a blank image, but we'll make things slightly more interesting by setting the initial image to be a Gaussian in the image plane, constant across all channels

```{code-cell}
# Gaussian parameters
kw = {
    "a": 1,
    "delta_x": 0.02,  # arcsec
    "delta_y": -0.01,
    "sigma_x": 0.02,
    "sigma_y": 0.01,
    "Omega": 20,  # degrees
}

# evaluate the Gaussian over the sky-plane, as np array
img_packed = utils.sky_gaussian_arcsec(
    coords.packed_x_centers_2D, coords.packed_y_centers_2D, **kw
)

# broadcast to (nchan, npix, npix)
img_packed_cube = np.broadcast_to(img_packed, (nchan, coords.npix, coords.npix)).copy()
# convert img_packed to pytorch tensor
img_packed_tensor = torch.from_numpy(img_packed_cube)
# insert into ImageCube layer
icube = images.ImageCube(coords=coords, nchan=nchan, cube=img_packed_tensor)
```

## Producing model visibilities

The interesting part of the NuFFT is that it will carry an image plane model all the way to the Fourier plane in loose visibilities, resulting in a model visibility array the same shape as the original visibility data.

```{code-cell}
vis_model_loose = nufft.forward(icube.forward())
print("Loose model visibilities from the NuFFT have shape {:}".format(vis_model_loose.shape))
print("The original loose data visibilities have shape {:}".format(data.shape))
```

By comparison, the {class}`~mpol.gridding.Gridder` object puts the visibilities onto a grid and exports a {class}`~mpol.datasets.GriddedDataset` object. These gridded data visibilities have the same dimensionality as the gridded model visibilities produced by the {class}`~mpol.fourier.FourierCube` layer

```{code-cell}
vis_model_gridded = flayer.forward(icube.forward())
print("Gridded model visibilities from FourierCube have shape {:}".format(vis_model_gridded.shape))
print("Gridded data visibilities have shape {:}".format(gridded_dset.vis_gridded.shape))
```

## Evaluating a likelihood function

As we discussed in the [Introduction to RML Imaging](../rml_intro.md) a likelihood function is used to to evaluate the probability of the data given a model and its parameters. 

### Preamble for a completely real dataset 

If we had a completely real dataset, for example, a bunch of values $\boldsymbol{Y}$ at various $\boldsymbol{X}$ locations and we wanted to fit a model of a line $M(x_i |\, \boldsymbol{\theta}) = m x_i + b$ with parameters $\boldsymbol{\theta} = \{m, b\}$, then the full likelihood is a multi-dimensional Gaussian

$$
\mathcal{L}(\boldsymbol{Y}|\,\boldsymbol{\theta}) = \frac{1}{[(2 \pi)^N \det \mathbf{\Sigma}]^{1/2}} \exp \left (- \frac{1}{2} \mathbf{R}^\mathrm{T} \mathbf{\Sigma}^{-1} \mathbf{R} \right )
$$

where $\boldsymbol{R} = \boldsymbol{Y} - M(\boldsymbol{X} |\, \boldsymbol{\theta})$ is a vector of residual visibilities and $\mathbf{\Sigma}$ is the covariance matrix of the data. The logarithm of the likelihood function is

$$
\ln \mathcal{L}(\boldsymbol{Y}|\,\boldsymbol{\theta}) = - \frac{1}{2} \left ( N \ln 2 \pi +  \ln \det \mathbf{\Sigma} + \mathbf{R}^\mathrm{T} \mathbf{\Sigma}^{-1} \mathbf{R} \right ).
$$

When considering independent data within the same channel, the covariance matrix is a diagonal matrix

$$
\mathbf{\Sigma} = \begin{bmatrix}
\sigma_1^2 & 0 & \ldots & 0 \\
0 & \sigma_2^2 & \ldots & 0 \\
\vdots & \vdots & \ddots & 0 \\
0 & 0 & 0 & \sigma_N^2
\end{bmatrix}
$$

and the logarithm of the likelihood can be reduced to the following expression

$$
\ln \mathcal{L}(\boldsymbol{Y}|\,\boldsymbol{\theta}) = - \frac{1}{2} \left ( N \ln 2 \pi +  \sum_i^N \sigma_i^2 + \chi^2(\boldsymbol{Y}|\,\boldsymbol{\theta}) \right )
$$

with 

$$
\chi^2(\boldsymbol{Y}|\,\boldsymbol{\theta}) = \sum_i^N \frac{(Y_i - M(x |\,\boldsymbol{\theta}))^2}{\sigma_i^2}.
$$

### Changes for complex-valued Fourier data

Evaluating a likelihood function for complex-valued Fourier data essentially follows the same rules, but with a few modifications. The first comes about because a complex data point actually contains more information than a single real data point, there two numbers (real and imaginary) instead of just one.
```{code-cell} More bits
This is why numpy and pytorch have a `complex128` data type, compared to a `float64`.
```

Within a given channel, Fourier data from sub-mm interferometric arrays like ALMA is well-characterized by independent Gaussian noise (the [cross-channel situation](https://github.com/MPoL-dev/MPoL/issues/18) is another story). 


where the $\chi^2$ is evaluated with complex-valued data and model components as

$$
\chi^2(\boldsymbol{V}|\,\boldsymbol{\theta}) = \sum_i^N \frac{|V_i - M_\mathcal{V}(u_i, v_i |\,\boldsymbol{\theta})|^2}{\sigma_i^2}.
$$

If the same image-plane model values are the same, then the calculation of the likelihood function should also be the same whether we use the {class}`mpol.fourier.NuFFT` to produce loose visibilities or we use the {class}`mpol.fourier.FourierCube` to compute gridded visibilities.

We'll test that now.

### "Loose" visibility log likelihood

```{code-cell}
 
```

### Gridded visibility log likelihood 


log_likelihood_gridded

log_likelihood



## Normalized negative log likelihood loss function

Most of the time, we will be working in situations where the $\sigma_i$ values of the dataset are assumed to be constant and we do not necessarily care about the absolute value of $\mathcal{L}(\boldsymbol{V}|\,\boldsymbol{\theta})$ but rather its relative value as a function of $\boldsymbol{\theta}$ (for example, as with posterior inference using MCMC). In these situations, we can further reduce the log likelihood function to a proportionality

$$
\ln \mathcal{L}(\boldsymbol{V}|\,\boldsymbol{\theta}) \propto - \frac{1}{2} \chi^2(\boldsymbol{V}|\,\boldsymbol{\theta}) .
$$

If we are simply optimizing a function, as with a simple parameter optimization, we only care about the value of $\boldsymbol{\theta}$ that maximizes the likelihood function---the constant of proportionality does not matter.

In an RML workflow, rather than talk about maximizing likelihood functions, we usually talk about minimizing loss functions. As described in the [Introduction to RML Imaging](../rml_intro.md), we will usually compile a target loss function as the sum of several individual loss functions: a (negative) log-likelihood loss function and several regularizers. In this application, the relative proportionality (strength) of each loss function or regularizer is important. If we use the form of $\chi^2$ discussed so far, then its absolute value will change significantly when we use more or less data. This means that in order to give similar relative regularization, the pre-factors of the other loss terms will also need to be changed in response. That is why in these applications we recommend using a normalized negative log likelihood loss function of the form

```{margin} Wrong for uncertainties
N.B. that you do not use this normalized form of the likelihood function for any quantification of the uncertainty on your model parameters. Because it has the wrong proportionality, its relative dependence on $\boldsymbol{\theta}$ will be different and therefore yield incorrect parameter uncertainties.
```

$$
L = \frac{1}{2 N_V} \sum_i^{N_V} \frac{|V_i - M_\mathcal{V}(u_i, v_i |\,\boldsymbol{\theta})|^2}{\sigma_i^2}
$$
similar to "reduced $\chi^2$," but where the extra factor of 2 in the denominator comes about because the visibility data is complex-valued. More details are available in {func}`mpol.losses.nll` and {func}`mpol.losses.nll_gridded`.



## timing tests -- possible with actual dataset
