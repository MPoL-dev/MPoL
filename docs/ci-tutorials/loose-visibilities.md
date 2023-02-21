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

Now let's put the NuFFT aside for a moment while we initialize the {class}`mpol.gridding.DataAverager` object and create an image for use in the forward model.

## Compared to the {class}`mpol.gridding.DataAverager` object

As before, we simply send the visibilities to the object and export a {class}`mpol.datasets.GriddedDataset`

```{code-cell}
averager = gridding.DataAverager(
    coords=coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data_re,
    data_im=data_im,
)

gridded_dset = averager.to_pytorch_dataset()
```

And we can initialize a {class}`mpol.fourier.FourierCube`

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
vis_model_loose = nufft(icube())
print("Loose model visibilities from the NuFFT have shape {:}".format(vis_model_loose.shape))
print("The original loose data visibilities have shape {:}".format(data.shape))
```

By comparison, the {class}`~mpol.gridding.Gridder` object puts the visibilities onto a grid and exports a {class}`~mpol.datasets.GriddedDataset` object. These gridded data visibilities have the same dimensionality as the gridded model visibilities produced by the {class}`~mpol.fourier.FourierCube` layer

```{code-cell}
vis_model_gridded = flayer(icube())
print("Gridded model visibilities from FourierCube have shape {:}".format(vis_model_gridded.shape))
print("Gridded data visibilities have shape {:}".format(gridded_dset.vis_gridded.shape))
```

## Evaluating a likelihood function

As we discussed in the [Introduction to RML Imaging](../rml_intro.md) a likelihood function is used to to evaluate the probability of the data given a model and its parameters.

### Preamble for a completely real dataset

If we had a dataset of only real values, for example, a bunch of values $\boldsymbol{Y}$ at various $\boldsymbol{X}$ locations, and we wanted to fit a model of a line $M(x_i |\, \boldsymbol{\theta}) = m x_i + b$ with parameters $\boldsymbol{\theta} = \{m, b\}$, then the full likelihood is a multi-dimensional Gaussian

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

```{margin} More bits
This is why both numpy and pytorch have a `complex128` data type, which stores real and imaginary components as `float64` values.
```
The likelihood function for complex-valued Fourier data with Gaussian uncertainties follows the same pattern, but with a few modifications. A complex data point actually contains more information than a single real data point, since there two numbers (real and imaginary) instead of just one.

Thankfully, as we pointed out in the [Introduction to RML Imaging](../rml_intro.md), the measurements of the real and imaginary components are independent. This simplifies life tremendously and means that we can write the joint likelihood function of the full complex-valued visibility dataset as the product of the likelihood function for the real visibilities and the likelihood function for the imaginary visibilities

$$
\mathcal{L}(\boldsymbol{V} |\,\boldsymbol{\theta} ) = \mathcal{L}(\boldsymbol{V}_\mathrm{Re} |\,\boldsymbol{\theta} ) \mathcal{L}(\boldsymbol{V}_\mathrm{Im} |\,\boldsymbol{\theta} ).
$$

Within a given channel, Fourier data from sub-mm interferometric arrays like ALMA is well-characterized by independent Gaussian noise (the [cross-channel situation](https://github.com/MPoL-dev/MPoL/issues/18) is another story). Therefore, we can follow the same simplifications as before to arrive at an expression for the log likelihood function of complex visibility data

```{margin} Full likelihood function
Note that this full form of $\ln \mathcal{L}(\boldsymbol{V} |\,\boldsymbol{\theta} )$ is what you'll want to use in any situation where you are doing parameter inference, care about the uncertainties on your parameters (e.g., an MCMC fit), and may adjust $\sigma_i$ values.
```
$$
\ln \mathcal{L}(\boldsymbol{V} |\,\boldsymbol{\theta} ) = - \left ( N \ln 2 \pi +  \sum_i^N \sigma_i^2 + \frac{1}{2} \chi^2(\boldsymbol{V}|\,\boldsymbol{\theta}) \right )
$$

Note than an extra factor of 2 appears in some places compared to the fully-real example. In this situation, the $\chi^2$ is either directly evaluated with complex-valued data and model components as

$$
\chi^2(\boldsymbol{V}|\,\boldsymbol{\theta}) = \sum_i^N \frac{|V_i - M(u_i, v_i |\,\boldsymbol{\theta})|^2}{\sigma_i^2}
$$

or is split into separate $\chi^2$ sums for the real and imaginary data

$$
\chi^2(\boldsymbol{V}|\,\boldsymbol{\theta}) = \sum_i^N \frac{(V_{\mathrm{Re},i} - M_\mathrm{Re}(u_i, v_i |\,\boldsymbol{\theta}))^2}{\sigma_i^2} + \sum_i^N \frac{(V_{\mathrm{Im},i} - M_\mathrm{Im}(u_i, v_i |\,\boldsymbol{\theta}))^2}{\sigma_i^2}.
$$

```{margin} Simplified likelihood function
You can use this form of $\ln \mathcal{L}(\boldsymbol{V} |\,\boldsymbol{\theta} )$ in any situation where you are doing parameter inference, care about the uncertainties on your parameters (e.g., an MCMC fit), but are keeping $\sigma_i$ values fixed.
```
Many times, we will be working in situations where the $\sigma_i$ values of the dataset are assumed to be constant. In these situations, we can further reduce the log likelihood function to a proportionality

$$
\ln \mathcal{L}(\boldsymbol{V}|\,\boldsymbol{\theta}) \propto - \frac{1}{2} \chi^2(\boldsymbol{V}|\,\boldsymbol{\theta}) .
$$

Though it's common to talk about likelihood functions as "the probability of the data" given model parameters, especially in a Bayesian context, the overall normalization of the likelihood function on its own [is not defined](https://hea-www.harvard.edu/AstroStat/aas227_2016/lecture1_Robinson.pdf). This means that the value of the likelihood function (and component parts, like $\chi^2$) can and will be different if the dataset has been binned, even though we might be using exactly the same model. This isn't a problem, though, because the important thing is that, regardless of normalization, the likelihood function will convey the same information (mean and uncertainty) about the model parameters, because this depends on the shape of $\mathcal{L}$ as a function of $\boldsymbol{\theta}$.

In a Bayesian context, the normalization of the posterior distribution is typically called the "Bayesian evidence." In evaluating this evidence, any normalization constant for the likelihood function would cancel out (e.g., see Equations 10 & 11 of [Hogg 2011](https://arxiv.org/abs/1205.4446)).

Now we'll evaluate the likelihood function using both the loose visibilities produced using the {class}`mpol.fourier.NuFFT` and the gridded visibilities produced using the {class}`mpol.fourier.FourierCube`.

### "Loose" visibility log likelihood

```{code-cell}
# convert data and weight to pytorch tensors for use in the calls
data_loose = torch.tensor(data)
weight_loose = torch.tensor(weight)

chisquare = losses.chi_squared(vis_model_loose, data_loose, weight_loose)
loglike = losses.log_likelihood(vis_model_loose, data_loose, weight_loose)
print("Chi squared", chisquare)
print("Log likelihood", loglike)
```

### Gridded visibility log likelihood

```{code-cell}
chisquare_gridded = losses.chi_squared_gridded(vis_model_gridded, gridded_dset)
loglike_gridded = losses.log_likelihood_gridded(vis_model_gridded, gridded_dset)
print("Chi squared gridded", chisquare_gridded)
print("Log likelihood gridded", loglike_gridded)
```

As we just discussed, it's OK that these evaluations are different between the loose and the gridded visibilities, even though we are using the exact same image plane model.


## Normalized negative log likelihood loss function

In an RML workflow, rather than talk about maximizing likelihood functions, we usually talk about minimizing loss functions. As described in the [Introduction to RML Imaging](../rml_intro.md), we will usually compile a target loss function $L$ as the sum of several individual loss functions: a (negative) log-likelihood loss function and several regularizers. If we are just optimizing $L$, we only care about the value of $\hat{\boldsymbol{\theta}}$ that minimizes the function $L$. The constant of proportionality does not matter, we only care that $L(\hat{\boldsymbol{\theta}}) < L(\hat{\boldsymbol{\theta}} + \epsilon)$, not by how much.

In this application, the *relative proportionality* (strength) of each loss function or regularizer is important. If we use the log likelihood function discussed so far, then its absolute value will change significantly when we use more or less data. This means that in order to give similar relative regularization, the pre-factors of the other loss terms will also need to be changed in response. This makes it hard to translate ballpark regularizer strengths from one dataset to another.

```{margin} Wrong for uncertainties
N.B. that you do not use this normalized form of the likelihood function for any quantification of the uncertainty on your model parameters. Because it has the wrong proportionality, its relative dependence on $\boldsymbol{\theta}$ will be different and therefore yield incorrect parameter uncertainties.
```
In these applications we recommend using a normalized negative log likelihood loss function of the form

$$
L_\mathrm{nll} = \frac{1}{2 N} \sum_i^{N} \frac{|V_i - M(u_i, v_i |\,\boldsymbol{\theta})|^2}{\sigma_i^2} =  \frac{1}{2 N} \chi^2(\boldsymbol{V}|\,\boldsymbol{\theta}),
$$

following [EHT-IV 2019](https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract). This formulation works because a "well-fit" model will have

$$
V_i \approx M(u_i, v_i |\,\boldsymbol{\theta})
$$

but will be wrong on average by the amount of noise $\epsilon_i$ present in $V_i$. Thus,

$$
\langle |V_i - M(u_i, v_i |\,\boldsymbol{\theta})|^2 \rangle = \sigma_{\mathrm{Re},i}^2 + \sigma_{\mathrm{Im},i}^2 = 2 \sigma_i^2
$$

This is the same reasoning that gives rise to the statistical apothegm "a well-fit model has a reduced $\chi^2_R \approx 1$" (one that has [many caveats and pitfalls](https://arxiv.org/abs/1012.3754)). In this case the extra factor of 2 in the denominator comes about because the visibility data and its noise are complex-valued.

The hope is that for many applications, the normalized negative log likelihood loss function will have a minimum value of $L(\hat{\boldsymbol{\theta}}) \approx 1$ for a well-fit model (regardless of the number of data points), making it easier to set the regularizer strengths *relative* to this value. Note that even this normalized loss won't be the same between an unbinned and binned dataset, though hopefully both will be on the order of $1$.

This loss function is implemented in {func}`mpol.losses.nll` and {func}`mpol.losses.nll_gridded`, and we can see the results here.

```{code-cell}
nll = losses.nll(vis_model_loose, data_loose, weight_loose)
print("Normalized log likelihood", nll)
```

```{code-cell}
nll_gridded = losses.nll_gridded(vis_model_gridded, gridded_dset)
print("Normalized log likelihood gridded", nll_gridded)
```
