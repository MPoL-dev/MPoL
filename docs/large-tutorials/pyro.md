---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [hide-cell]

%matplotlib inline
%run notebook_setup
```

# Parametric Inference with Pyro

In all of the tutorials thus far, we have used MPoL to optimize non-parametric image plane models, i.e., collections of pixels. However, there may be instances where the astrophysical source morphology is simple enough at the resolution of the data such that an investigator might wish to fit a parametric model to the data. In the protoplanetary disk field, there is a long history of parametric model fits to data. The simplest example of this would be an elliptcial Gaussian fit through CASA's [uvmodelfit](https://casadocs.readthedocs.io/en/stable/api/tt/casatasks.manipulation.uvmodelfit.html), while a more complex example might be the [Galario](https://mtazzari.github.io/galario/) package. While non-paramtetric models tend to get all of the attention in this era of Big Data, well-constructed parametric models can still prove useful thanks to their interpretability and role in Bayesian inference.

In this tutorial, we will explore how we can use MPoL with a probabilistic programming language called [Pyro](https://pyro.ai/) to perform parametric model fitting with a continuum protoplanetary disk dataset and derive posterior probability distributions of the model parameters. One major advantage of using MPoL + Pyro to do parametric model fitting compared to existing packages is that posterior gradient information is naturally provided by PyTorch's autodifferentiation capabilities. This, coupled with the industry-grade inference algorithms provided by Pyro, makes it computationally efficient to explore posterior probability distributions with dozens or even hundreds of parameters--something that would be impractical using classical MCMC algorithms.

+++

## MPoL and models

Before we discuss the specifics of the parametric disk model, let's take a moment to do a high-level review of what a typical MPoL model might look like. 

### Non-parametric models 
Let's start by considering the architecture of the simplest possible skeleton non-parametric RML model


```{mermaid} ../_static/mmd/src/ImageCube.mmd
```

When we say that a model is non-parametric we generally mean that the number of parameters of the model is vast (potentially infinite) and can grow to encapsulate more detail if needed. A classic example is something like a spline or a Gaussian process, but in our case we are using a large number of discrete pixel fluxes to represent an image.

We can see the definition of the "non-parametric" image parameters in the Pytorch layer

```
self.cube = nn.Parameter(
    torch.full(
        (self.nchan, self.coords.npix, self.coords.npix),
        fill_value=0.0,
        requires_grad=True,
        dtype=torch.double,
    )
)
```
The `nn.Parameter` call tells Pytorch that the `cube` tensor should be varied during optimization of the model. The `cube` tensor is effectively the set of parameters for the "non-parametric" model.

+++

We can consider the architecture of the {class}`mpol.precomposed.SimpleNet` as a more practical extension 


```{mermaid} ../_static/mmd/src/SimpleNet.mmd
```

The functionality of the {class}`mpol.precomposed.SimpleNet` is similar to the skeleton model, but we've shifted the base parameterization from the {class}`mpol.images.ImageCube` to the {class}`mpol.images.BaseCube` (so that pixel flux values are non-negative) and we've included a small convolution kernel (through {class}`mpol.images.HannConvCube`) so that high-spatial-frequency noise is supressed. In this framework, the `nn.Parameter`s are instantiated on the {class}`~mpol.images.BaseCube` and the {class}`~mpol.images.ImageCube` becomes a pass-through layer.

In both of these cases, the key functionality provided by the MPoL package is the {class}`mpol.fourier.FourierCube` layer that translates a model image into the visibility plane. From the perspective of the {class}`~mpol.fourier.FourierCube`, it doesn't care how the model image was produced, it will happily translate image pixels into visibility values using the FFT.

### Parametric models

By contrast to a non-parametric model, a *parametric* model is one that has a (finite) set of parameters (generally decoupled from the size of the data) and can be easily used to make future predictions of the data, usually in a functional form. For example, a cubic function and its coefficients would be considered a parametric model. For a radio astronomy example, you can think of the {class}`~mpol.images.BaseCube` and {class}`mpol.images.HannConvCube` layers as being replaced by a parametric disk model, which we'll call `DiskModel`. This parametric model would specify pixel brightness as a function of position based upon model parameters, and would feed directly into the {class}`~mpol.images.ImageCube` pass-through layer.

```{mermaid} ../_static/mmd/src/Parametric.mmd
```

Before ALMA, it was common in the protoplanetary disk field to fit parametric models (e.g., elliptical Gaussians, one or two axisymmetric rings, etc...) to interferometric observations to derive source properties like size and inclination. The spatial resolution afforded by the ALMA long-baseline campaign rendered many of these simple parametric models inadequate. Suddenly, rich substructure in the forms of rings, gaps, and spirals was visible in dust continuum images and, except for a few exceptions we'll discuss in a second, these morphologies were too complex to neatly capture with simple model parameterizations.

This spurred a major shift from parametric, visibility-based analyses to image-based analysis (including our own MPoL efforts). For axisymmetric sources, visibility-based analysis is still viable thanks to the development of novel non-parametric 1D models like [frank](https://discsim.github.io/frank/), which are capable of super-resolution compared to image-based methods like CLEAN.

In our opinion, the two (linked) reasons that parametric model fitting has fallen out of favor in the protoplanetary disk field are 

1. ALMA data are sufficiently high quality that many model parameters are required to accurately describe disk emission
2. standard sampling algorithms used for Bayesian inference do not perform well in high dimensional parameter spaces

As we hinted at, the MPoL + Pyro + PyTorch framework will help us out on point #2, such that we might be able to explore more detailed models with larger numbers of parameters.

This point of this tutorial isn't to say that actually everyone should switch back to using parametric models. But rather, that, with the industry-grade machinery of probabilistic programming languages and autodifferentiation, there may be situations where parametric models are still useful.

+++

## DSHARP AS 209 dataset

For this tutorial we'll use the ALMA DSHARP dust continuum observations of the AS 209 protoplanetary disk. The data reduction is described in [Andrews et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..41A/abstract) and the primary analysis is described in [Guzmán et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..48G/abstract).

The original measurement sets from the DSHARP program are available in measurement set format from the ALMA project pages (e.g., [NRAO](https://bulk.cv.nrao.edu/almadata/lp/DSHARP/)). To save some boilerplate code and computation time for the purposes of this tutorial, we have extracted the visibilities from this measurement set, performed a few averaging and weight scaling steps, and uploaded the processed dataset to Zenodo. It can be downloaded here. The full set of pre-processing commands are available in the mpoldatasets package.

```{code-cell} ipython3
fname = download_file(
    "https://zenodo.org/record/7732834/files/AS209_continuum_averaged.asdf",
    cache=True,
    pkgname="mpol",
)
```

```{code-cell} ipython3
# load extracted visibilities from npz file
d = np.load(fname)
uu = d["uu"]
vv = d["vv"]
weight = d["weight"]
data = d["data"]
```

Let's make some diagnostic images, to make sure we've loaded the data correctly.

Beam and Dirty image using the DirtyImager. More details in tutorial.

In their DSHARP paper, Guzmán et al. 2018 noted the striking azimuthal symmetry of the AS 209 disk. This motivated them to develop and fit a 1D surface brightness profile $I(r)$ using a series of concentric Gaussian rings of the form

$$
I(r) = \sum_{i=0}^N A_i \exp \left (- \frac{(r - r_i)^2}{2 \sigma_i^2} \right).
$$

The axisymmetry of the model allowed them to use the Hankel transform to compute the visibility function $\mathcal{V}$ corresponding to a given $I(r)$. The Hankel transform also plays a key role in non-parametric 1D methods like `frank`. Then, Guzmán et al. 2018 evaluated the probability of the data given the model visibilities using a likelihood function and assigned prior probability distributions to their model parameters. They used the [emcee](https://emcee.readthedocs.io/) MCMC ensemble sampler to sample the posterior distribution of the parameters and thus infer the surface brightness profile $I(r)$. 

In what follows we will use Pyro and the MPoL framework to implement the same concentric Gaussian ring model as Guzmán et al. 2018 and (hopefully) verify that we obtain the same result. But, we should note that because MPoL uses the 2D FFT to perform the Fourier Transform, we do not need to assume an axisymmetric model. This may be beneficial when fitting disk morphologies that are not purely axisymmetric.

## Introduction to Probabilistic Programming Languages

Many astronomers traditionally follow an MCMC analysis pathway similar to Guzmán et al. 2018: they write custom code to implement their model, calculate their likelihood function and priors, and then use an MCMC package like `emcee` to sample the posterior. 

[Probabilistic programming languages](https://en.wikipedia.org/wiki/Probabilistic_programming) (PPLs)
are by no means a recent invention, but have in recent years become much more powerful and scientifically capable thanks to the integration of autodifferentiation and advanced sampling methodologies that use gradient information. In our own subfield, we are most familiar with the [exoplanet](https://docs.exoplanet.codes/en/latest/) codebase, built on PyMC3; however, a quick search on ADS demonstrates that probabilistic programming languages have seen greater usage by astronomers in the past decade across a variety of subfields. 

Simply put, PPLs are frameworks that help users build statistical models and then infer/optimize the parameters of those models conditional on some dataset. PPLs usually have their own learning curve that requires familiarizing oneself with the syntax of the language and the mechanics of building models; once the learning curve is climbed, however, PPLs have the potential to be incredibly powerful inference tools.

[Pyro](https://pyro.ai/) is the main PPL built on PyTorch, so that is what we will use in this tutorial. In what follows we'll try to explain the relevant parts of Pyro that you'll need to get started, but a full introduction to Pyro and PPLs is beyond the scope of this tutorial. If you are interested, we recommend you see the following resources: 

* [Introduction to Pyro](http://pyro.ai/examples/intro_long.html)
* [Bayesian Regression - Introduction](http://pyro.ai/examples/bayesian_regression.html)

The Pyro [examples](http://pyro.ai/examples/index.html) page and [documentation](https://docs.pyro.ai/en/stable/) have much more information that can help you get started.

We also recommend reading Gelman et al. 2020's paper on [Bayesian Workflow](https://arxiv.org/abs/2011.01808). It contains very useful advice on structuring a large and complex Bayesian data analysis problem and will no doubt save you time when constructing your own models.

If you are new to Bayesian analysis in general, we recommend that you put this tutorial aside for a moment and review some introductory resources like [Eadie et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230204703E/abstract) and references therein.


## Building a parametric disk model 

There are many ways to build a Pyro model. In this tutorial we will take a class-based approach and use the [PyroModule](http://pyro.ai/examples/modules.html) construct, but models can just as easily be built using function definitions (for [example](http://pyro.ai/examples/intro_long.html#Models-in-Pyro)).

```{code-cell} ipython3
def plot_1D_profile(rs, Is):
    fig, ax = plt.subplots(nrows=1, figsize=(4,4))
    ax.plot(rs, Is)
    ax.set_xlabel("r [au]")
    ax.set_ylabel(r"I [Jy/$\mathrm{arcsec}^2$]")
    fig.subplots_adjust(left=0.2, right=0.8)

    return fig
```

```{code-cell} ipython3
def compare_dirty_model_resid(model_real, model_imag, sky_cube, robust=0.0):

    # convert to numpy 
    model_real = model_real.detach().numpy()
    model_imag = model_imag.detach().numpy()

    data_real = np.real(loaddata.data)
    data_imag = np.imag(loaddata.data)
    resid_real = data_real - model_real 
    resid_imag = data_imag - model_imag

    img_dirty, _ = make_dirty_image(data_real, data_imag)
    img_model, _ = make_dirty_image(model_real, model_imag)
    img_resid, _ = make_dirty_image(resid_real, resid_imag)

    # determine the plot dimensions
    xx = 12 # in
    cax_width = 0.2 # in 
    cax_sep = 0.1 # in
    mmargin = 1.2
    lmargin = 0.7
    rmargin = 0.7
    tmargin = 0.3
    bmargin = 0.5
    
    npanels = 4
    # the size of image axes + cax_sep + cax_width
    block_width = (xx - lmargin - rmargin - mmargin * (npanels - 1) )/npanels
    ax_width = block_width - cax_width - cax_sep
    ax_height = ax_width 
    yy = bmargin + ax_height + tmargin

    fig = plt.figure(figsize=(xx, yy))
    ax = []
    cax = []
    for i in range(npanels):
        ax.append(fig.add_axes([(lmargin + i * (block_width + mmargin))/xx, bmargin/yy, ax_width/xx, ax_height/yy]))
        cax.append(fig.add_axes([(lmargin + i * (block_width + mmargin) + ax_width + cax_sep)/xx, bmargin/yy, cax_width/xx, ax_height/yy]))

    chan = 0

    im = ax[0].imshow(img_dirty[chan], **kw)
    ax[0].set_title("dirty image")
    cbar = plt.colorbar(im, cax=cax[0])
    cbar.set_label(r"Jy/$\mathrm{arcsec}^2$")
    
    im = ax[1].imshow(sky_cube[chan], **kw)
    ax[1].set_title("model image")
    cbar = plt.colorbar(im, cax=cax[1])
    cbar.set_label(r"Jy/$\mathrm{arcsec}^2$")
    
    im = ax[2].imshow(img_model[chan], **kw)
    ax[2].set_title("model vis imaged")
    cbar = plt.colorbar(im, cax=cax[2])
    cbar.set_label(r"Jy/$\mathrm{arcsec}^2$")
    
    im = ax[3].imshow(img_resid[chan], **kw)
    ax[3].set_title("residual vis imaged")
    cbar = plt.colorbar(im, cax=cax[3])
    cbar.set_label(r"Jy/$\mathrm{arcsec}^2$")
    
    for a in ax:
        a.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
        a.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
    
    return fig 
```

```{code-cell} ipython3
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm
import loaddata

from mpol import coordinates, gridding

# use MPoL to make a dirty image 
coords = coordinates.GridCoords(cell_size=0.005, npix=800)
kw = {"origin": "lower", "interpolation": "none", "extent": coords.img_ext}

def get_pm(tensor, log10=False):
    t = tensor.detach().numpy()
    if log10:
        t = np.power(10.0, t)

    median, low, high = t

    minus = median - low
    plus = high - median

    return [median, plus, minus]


def pprint_quantiles(guide):
    """
    Get quantiles from a guide and pretty print output to compare to Guzman.
    """

    quantiles = guide.quantiles([0.5, 0.16, 0.84])

    A_0 = "A_0: {0:.2f} +/- {1:.2f}/{2:.2f} Jy/arcsec^2".format(
        *get_pm(quantiles["disk.log_A_0"], log10=True)
    )
    sigma_0 = "sigma_0: {0:.2f} +/- {1:.2f}/{2:.2f} au".format(
        *get_pm(quantiles["disk.log_sigma_0"], log10=True)
    )

    return_list = [A_0, sigma_0]

    ring_names = ["B15", "B27", "B41", "B74", "B92", "B120", "B140"]

    nrings = len(quantiles["disk.log_ring_amplitudes"][0])

    lra = get_pm(quantiles["disk.log_ring_amplitudes"] - quantiles["disk.log_A_0"][0], log10=True)
    lrm = get_pm(quantiles["disk.ring_means"])
    lrs = get_pm(quantiles["disk.log_ring_sigmas"], log10=True)

    for i in range(nrings):
        s = []
        s.append("\nRing {}".format(ring_names[i]))
        s.append(
            "A normed: {0:.3f} +/- {1:.3f}/{2:.3f} Jy/arcsec^2".format(
                lra[0][i], lra[1][i], lra[2][i]
            )
        )
        s.append(
            "mu: {0:.2f} +/- {1:.2f}/{2:.2f} au".format(lrm[0][i], lrm[1][i], lrm[2][i])
        )
        s.append(
            "sigma: {0:.2f} +/- {1:.2f}/{2:.2f} au".format(
                lrs[0][i], lrs[1][i], lrs[2][i]
            )
        )

        return_list += s

    return_list.append("x_centroid: {0:.4f} +/- {1:0.4f}/{2:0.4f} arcsec".format(*get_pm(quantiles["disk.x_centroid"])))
    return_list.append("y_centroid: {0:.4f} +/- {1:0.4f}/{2:0.4f} arcsec".format(*get_pm(quantiles["disk.y_centroid"])))


    print("\n".join(return_list))



def make_dirty_image(data_real, data_imag, robust=-0.5):
    """
    Make a plot of the dirty beam and dirty image (in units of Jy/arcsec^2).
    """

    imager = gridding.DirtyImager(
        coords=coords,
        uu=loaddata.uu,
        vv=loaddata.vv,
        weight=loaddata.weight,
        data_re=data_real,
        data_im=data_imag,
    )

    return imager.get_dirty_image(weighting="briggs", robust=robust, unit="Jy/arcsec^2")
```

build and heavily comment the disk model, including MPoL geometry routines, deterministic statements

## Parameter inference with Stochastic Variational Inference (SVI)
run SVI inference loop on GPU
analyze samples
explore MultiNormal fits to see if posterior changes

## Parameter inference with MCMC and Hamiltonian Monte Carlo
run HMC loop on GPU and analyze samples
show scatter in 1D profile as draws or movie

```{code-cell} ipython3

```
