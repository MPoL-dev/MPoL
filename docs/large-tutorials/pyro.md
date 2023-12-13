---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
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

In all of the tutorials thus far, we have used MPoL to optimize non-parametric image plane models, i.e., collections of pixels. However, there may be instances where the astrophysical source morphology is simple enough at the resolution of the data such that an investigator might wish to fit a parametric model to the data. In the protoplanetary disk field, there is a long history of parametric model fits to data. The simplest example of this would be an elliptical Gaussian fit through CASA's [uvmodelfit](https://casadocs.readthedocs.io/en/stable/api/tt/casatasks.manipulation.uvmodelfit.html), while a more complex example might be the [Galario](https://mtazzari.github.io/galario/) package. While non-parametric models tend to get all of the attention in this era of Big Data, well-constructed parametric models can still prove useful thanks to their interpretability and role in Bayesian inference.

In this tutorial, we will explore how we can use MPoL with a probabilistic programming language called [Pyro](https://pyro.ai/) to perform parametric model fitting with a continuum protoplanetary disk dataset and derive posterior probability distributions of the model parameters. One major advantage of using MPoL + Pyro to do parametric model fitting compared to existing packages is that posterior gradient information is naturally provided by PyTorch's autodifferentiation capabilities. This, coupled with the industry-grade inference algorithms provided by Pyro, makes it computationally efficient to explore posterior probability distributions with dozens or even hundreds of parameters--something that would be impractical using classical MCMC algorithms.

In this tutorial, we will use [Stochastic Variational Inference](http://pyro.ai/examples/svi_part_i.html) algorithms to obtain the posterior distribution of the model parameters. These algorithms are quick to implement in Pyro and--important for this tutorial--quick to run. Pyro also has full support for MCMC algorithms like Hamiltonian Monte Carlo and the No U-Turn Sampler (NUTS) ([for example](http://pyro.ai/examples/bayesian_regression_ii.html#HMC)) that are relatively straightforward to use in an extension from this model. However, because their run times are significantly longer than SVI algorithms, more computational resources are needed beyond the scope of this tutorial.

If the following output says `Using cuda`, then this tutorial was executed on a GPU. We found that it took about 5 minutes to converge the SVI, which is pretty exciting. You may be able to run this on CPU-only machine, but expect the runtime to take significantly longer. You may want to shorten the number of iterations and reduce the number of predictive samples to get a sense that the routine will in fact execute, but be aware that your solution may not fully converge.

```{code-cell} ipython3
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')                   
else:                                                       
    device = torch.device('cpu')   

print(f"Using {device}.")    
    
```

```{code-cell} ipython3
# import arviz now, to check UTF-8 loading issue.
import arviz as az
```

## MPoL and models

Before we discuss the specifics of the parametric disk model, let's take a high-level look at what makes up an MPoL model.

### Non-parametric models 
Let's start by considering the architecture of the simplest possible skeleton non-parametric RML model

```{mermaid} ../_static/mmd/src/ImageCube.mmd
```

When we say that a model is non-parametric we generally mean that the number of parameters of the model is vast (potentially infinite) and can grow to encapsulate more detail if needed. A classic example is something like a spline or a Gaussian process, but in our case we are using a large number of discrete pixel fluxes to represent an image.

We can see the definition of these "non-parametric" image parameters in the Pytorch layer

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
The `nn.Parameter` call tells Pytorch that the `cube` tensor containing the image pixels should be varied during optimization.

+++

We can consider the architecture of the {class}`mpol.precomposed.SimpleNet` as a more practical extension 

```{mermaid} ../_static/mmd/src/SimpleNet.mmd
```

The functionality of the {class}`mpol.precomposed.SimpleNet` is similar to the skeleton model, but we've shifted the base parameterization from the {class}`mpol.images.ImageCube` to the {class}`mpol.images.BaseCube` (so that pixel flux values are non-negative) and we've included a small convolution kernel (through {class}`mpol.images.HannConvCube`) so that high-spatial-frequency noise is suppressed. In this framework, the `nn.Parameter`s are instantiated on the {class}`~mpol.images.BaseCube` and the {class}`~mpol.images.ImageCube` becomes a pass-through layer.

In both of these cases, the key functionality provided by the MPoL package is the {class}`mpol.fourier.FourierCube` layer that translates a model image into the visibility plane. From the perspective of the {class}`~mpol.fourier.FourierCube`, it doesn't care how the model image was produced, it will happily translate image pixels into visibility values using the FFT.

### Parametric models

By contrast to a non-parametric model, a *parametric* model is one that has a (finite) set of parameters (generally decoupled from the size of the data) and can be easily used to make future predictions of the data, usually in a functional form. For example, a cubic function and its coefficients would be considered a parametric model. For a radio astronomy example, you can think of the {class}`~mpol.images.BaseCube` and {class}`mpol.images.HannConvCube` layers as being replaced by a parametric disk model, which we'll call `DiskModel`. This parametric model would specify pixel brightness as a function of position based upon model parameters, and would feed directly into the {class}`~mpol.images.ImageCube` pass-through layer.

```{mermaid} ../_static/mmd/src/Parametric.mmd
```

Before ALMA, it was common in the protoplanetary disk field to fit parametric models (e.g., elliptical Gaussians, one or two axisymmetric rings, etc...) to interferometric observations to derive source properties like size and inclination. The spatial resolution afforded by the ALMA long-baseline campaign rendered many of these simple parametric models inadequate. Suddenly, rich substructure in the forms of rings, gaps, and spirals was visible in dust continuum images and, except for a few exceptions we'll discuss in a second, these morphologies were too complex to neatly capture with simple model parameterizations.

This spurred a major shift from parametric, visibility-based analyses to image-based analysis (including our own MPoL efforts). Visibility-based analysis is still viable, but with modern datasets it must often be more sophisticated. For example, non-parametric 1D models like [frank](https://discsim.github.io/frank/) are capable of super-resolution compared to image-based methods like CLEAN for axisymmetric sources.

In our opinion, the two (linked) reasons that parametric model fitting has fallen out of favor in the protoplanetary disk field are 

1. ALMA data are sufficiently high quality that many model parameters are required to accurately describe disk emission
2. standard sampling algorithms used for Bayesian inference do not perform well in high dimensional parameter spaces

As we hinted at, the MPoL + Pyro + PyTorch framework will help us out on point #2, such that we might be able to explore more detailed models with larger numbers of parameters.

The point of this tutorial isn't to say that everyone should switch back to using parametric models. But rather that with the industry-grade machinery of probabilistic programming languages and autodifferentiation, there may be situations where parametric models are still useful.

+++

## DSHARP AS 209 dataset

For this tutorial we'll use the ALMA DSHARP dust continuum observations of the AS 209 protoplanetary disk. The data reduction is described in [Andrews et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..41A/abstract) and the primary analysis is described in [Guzmán et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..48G/abstract).

The original measurement sets from the DSHARP program are available in measurement set format from the ALMA project pages (e.g., [NRAO](https://bulk.cv.nrao.edu/almadata/lp/DSHARP/)). To save some boilerplate code and computation time for the purposes of this tutorial, we have extracted the visibilities from this measurement set, performed a few averaging and weight scaling steps, and uploaded the processed dataset to a Zenodo [repository](https://zenodo.org/record/7732834#.ZBCKAexKhhE) as an [asdf](https://asdf.readthedocs.io/en/stable/index.html) file. The full set of pre-processing commands are available in the [mpoldatasets package](https://github.com/MPoL-dev/mpoldatasets/tree/main/products/AS209-DSHARP-continuum-averaged). Let's download the file and extract the visibilities

```{code-cell} ipython3
from astropy.utils.data import download_file

fname = download_file(
    "https://zenodo.org/record/7732834/files/AS209_continuum_averaged.asdf",
    cache=True,
    pkgname="mpol",
)
```

```{code-cell} ipython3
import asdf 

# load extracted visibilities from asdf file 
d = asdf.open(fname)
uu = d["uu"]
vv = d["vv"]
weight = d["weight"]
data = d["data"]    
```

Let's use the MPoL {class}`~mpol.gridding.DirtyImager` to make some diagnostic images, to make sure we've loaded the data correctly.

```{code-cell} ipython3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

from mpol import coordinates, gridding

# settle on an image size that we'll use throughout the tutorial
coords = coordinates.GridCoords(cell_size=0.005, npix=800)
kw = {"origin": "lower", "interpolation": "none", "extent": coords.img_ext, "cmap":"inferno"}


def make_dirty_image(data_real, data_imag, robust=-0.5):
    """
    Make a plot of the dirty beam and dirty image (in units of Jy/arcsec^2).
    
    Args:
        data_real (numpy array): real components of visibilities
        data_imag (numpy array): imaginary components of visibilities
        robust (float): the Briggs robust parameter
        
    Returns:
        beam, image numpy arrays
    """

    imager = gridding.DirtyImager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_real,
        data_im=data_imag,
    )

    return imager.get_dirty_image(weighting="briggs", robust=robust, unit="Jy/arcsec^2")
```

```{code-cell} ipython3
img, beam = make_dirty_image(np.real(data), np.imag(data))

# set plot dimensions
xx = 8 # in
cax_width = 0.2 # in 
cax_sep = 0.1 # in
mmargin = 1.2
lmargin = 0.7
rmargin = 0.7
tmargin = 0.3
bmargin = 0.5

npanels = 2
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

# single-channel image cube    
chan = 0

im_beam = ax[0].imshow(beam[chan], **kw)
cbar = plt.colorbar(im_beam, cax=cax[0])
ax[0].set_title("beam")
# zoom in a bit
r = 0.3
ax[0].set_xlim(r, -r)
ax[0].set_ylim(-r, r)

im = ax[1].imshow(img[chan], **kw)
ax[1].set_title("dirty image")
cbar = plt.colorbar(im, cax=cax[1])
cbar.set_label(r"Jy/$\mathrm{arcsec}^2$")

for a in ax:
    a.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
    a.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
```

In their DSHARP paper, Guzmán et al. 2018 noted the striking azimuthal symmetry of the AS 209 disk. This motivated them to develop and fit a 1D surface brightness profile $I(r)$ using a series of concentric Gaussian rings of the form

$$
I(r) = \sum_{i=0}^N A_i \exp \left (- \frac{(r - r_i)^2}{2 \sigma_i^2} \right).
$$

The axisymmetry of the model allowed them to use the Hankel transform to compute the visibility function $\mathcal{V}$ corresponding to a given $I(r)$. The Hankel transform also plays a key role in non-parametric 1D methods like  [frank](https://discsim.github.io/frank/). Guzmán et al. 2018 evaluated the probability of the data given the model visibilities using a likelihood function and assigned prior probability distributions to their model parameters. They used the [emcee](https://emcee.readthedocs.io/) MCMC ensemble sampler to sample the posterior distribution of the parameters and thus infer the surface brightness profile $I(r)$. 

In what follows we will use Pyro and the MPoL framework to implement the same concentric Gaussian ring model as Guzmán et al. 2018 and (hopefully) verify that we obtain the same result. But, we should note that because MPoL uses the 2D FFT to perform the Fourier Transform, we do not need to assume an axisymmetric model. This may be beneficial when fitting disk morphologies that are not purely axisymmetric.

+++

## Introduction to Probabilistic Programming Languages

Many astronomers usually follow an MCMC analysis pathway similar to Guzmán et al. 2018: they write custom code to implement their model, calculate their likelihood function and priors, and then use an MCMC package like `emcee` to sample the posterior. 

[Probabilistic programming languages](https://en.wikipedia.org/wiki/Probabilistic_programming) (PPLs)
are by no means a recent invention, but have in recent years become much more powerful and scientifically capable thanks to the integration of autodifferentiation and advanced sampling methodologies that use gradient information. In our own subfield, we are most familiar with the [exoplanet](https://docs.exoplanet.codes/en/latest/) codebase, built on PyMC3; however, a quick search on ADS demonstrates that probabilistic programming languages have seen greater usage by astronomers in the past decade across a variety of subfields. 

Simply put, PPLs are frameworks that help users build statistical models and then infer/optimize the parameters of those models conditional on some dataset. PPLs usually have their own learning curve that requires familiarizing oneself with the syntax of the language and the mechanics of building models; once the learning curve is climbed, however, PPLs have the potential to be incredibly powerful inference tools.

[Pyro](https://pyro.ai/) is the main PPL built on PyTorch, so that is what we will use in this tutorial. In what follows we'll try to explain the relevant parts of Pyro that you'll need to get started, but a full introduction to Pyro and PPLs is beyond the scope of this tutorial. If you are interested, we recommend you see the following resources: 

* [Introduction to Pyro](http://pyro.ai/examples/intro_long.html)
* [Bayesian Regression - Introduction](http://pyro.ai/examples/bayesian_regression.html)

The Pyro [examples](http://pyro.ai/examples/index.html) page and [documentation](https://docs.pyro.ai/en/stable/) have much more information that can help you get started.

```{margin} New to Bayes
If you are new to Bayesian analysis in general, we recommend that you put this tutorial aside for a moment and review some introductory resources like [Eadie et al. 2023](https://ui.adsabs.harvard.edu/abs/2023arXiv230204703E/abstract) and references therein.
```

We also recommend reading Gelman et al. 2020's paper on [Bayesian Workflow](https://arxiv.org/abs/2011.01808). It contains very useful advice on structuring a large and complex Bayesian data analysis problem and will no doubt save you time when constructing your own models.

+++

## Building a parametric disk model 

There are many ways to build a Pyro model. In this tutorial we will take a class-based approach and use the [PyroModule](http://pyro.ai/examples/modules.html) construct, but models can just as easily be built using function definitions (for [example](http://pyro.ai/examples/intro_long.html#Models-in-Pyro)).

```{code-cell} ipython3
from torch import nn
from mpol import geometry, gridding, images, fourier, utils
from mpol.constants import deg

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample, pyro_method
```

First, we'll define a class that we'll call `PyroDisk`. This class defines Guzmán et al. 2018's ringed model using the Pyro PPL and produces an image.

```{code-cell} ipython3
class PyroDisk(PyroModule):
    r"""
    This routine returns an image.
    """

    def __init__(
        self,
        coords=None,
        nchan=1,
        distance=None,
    ):
        super().__init__()

        self.coords = coords
        self.nchan = nchan

        # observer-frame coordinates
        YY = torch.as_tensor(self.coords.packed_x_centers_2D.copy())
        XX = torch.as_tensor(self.coords.packed_y_centers_2D.copy())
        self.register_buffer("YY", YY)
        self.register_buffer("XX", XX)
        # This mashup is because of the way we define the coordinate system for orbital elements.
        # YY points north
        # XX points east

        # setup geometric parameters

        # the model is axisymmetric, so argument of periastron is degenerate. We set this to 0 and 
        # do not sample in it.
        self.omega = 0 

        # we have a reasonably good guess as to these orientations from inspection of the 
        # dirty image and so Normal priors are fine. 
        # If we were very uncertain about these parameters, it might make sense using 
        # the Von Mises distribution for the angles like omega, incl, and Omega
        # https://docs.pyro.ai/en/stable/distributions.html?highlight=constraints#vonmises
        # https://en.wikipedia.org/wiki/Von_Mises_distribution
        self.incl = PyroSample(dist.Normal(35. * deg, 5. * deg))
        self.Omega = PyroSample(dist.Normal(85.0 * deg, 10.0 * deg))
        
        # to treat parameters as fixed, simply assign them as torch tensors
        # for example,
        # self.x_centroid = torch.as_tensor(x_centroid)  # arcsec
        # self.y_centroid = torch.as_tensor(y_centroid)  # arcsec

        # otherwise, define latent random variables using PyroSample
        # and a distribution object
        self.x_centroid = PyroSample(dist.Normal(0.0, 3e-3)) # arcsec
        self.y_centroid = PyroSample(dist.Normal(0.0, 3e-3)) # arcsec

        self.distance = torch.as_tensor(distance)  # pc

        # Define a 1D radial grid for evaluating the 1D intensity profile
        self.R = torch.linspace(0.0, torch.max(torch.concat([XX, YY])), steps=400) * self.distance

       # central Gaussian envelope
        self.log_A_0 = PyroSample(dist.Normal(0.0, 0.3))
        self.log_sigma_0 = PyroSample(dist.Normal(0.7, 0.1))
    
        # list of Gaussian parameters
        # ring means from Huang et al. 2018a.
        ring_means = torch.as_tensor(np.array([14., 28., 41., 74., 99., 120., 141.]))
        self.nrings = torch.as_tensor(len(ring_means))

        self.log_ring_sigmas = PyroSample(
            dist.Normal(0.8, 0.3).expand([self.nrings]).to_event(1)
        )

        self.log_ring_amplitudes = PyroSample(
            dist.Normal(-1.0, 0.5).expand([self.nrings]).to_event(1)
        )

        # we set the mean of the Normal prior on the ring means to the values from Huang
        self.ring_means = PyroSample(dist.Normal(ring_means, 10.0).to_event(1))
            

    @pyro_method
    def _Gaussian(self, r, A_i, r_i, sigma_i):
        r"""
        Evaluate a Gaussian ring of the form

        .. math::

            f(r) = A_i \exp \left(- \frac{(r - r_i)^2}{2 \sigma_i^2} \right)
        """
        return A_i * torch.exp(-0.5 * (r - r_i) ** 2 / sigma_i**2)

    @pyro_method
    def intensity_profile(self, r):
        r"""
        Evaluate the intensity profile.
        """
        I = torch.zeros_like(r)

        # evaluate the central Gaussian
        A_0 = torch.pow(10.0, self.log_A_0)
        r_0 = 0.0
        sigma_0 = torch.pow(10.0, self.log_sigma_0)

        I += self._Gaussian(r, A_0, r_0, sigma_0)

        # evaluate the rings
        for i in range(self.nrings):
            A_i = torch.pow(10.0, self.log_ring_amplitudes[i])
            r_i = self.ring_means[i]
            sigma_i = torch.pow(10.0, self.log_ring_sigmas[i])
            I += self._Gaussian(r, A_i, r_i, sigma_i)

        return I

    def forward(self):
        # take 2D coords object and project it to 2D frame
        # units of arcseconds
        x_warped, y_warped = geometry.observer_to_flat(
            self.XX, self.YY, omega=self.omega, incl=self.incl, Omega=self.Omega
        )

        # apply centroid offset
        xx = x_warped - self.x_centroid
        yy = y_warped - self.y_centroid

        # convert x,y to radial coordinates and then to AU
        rr = torch.hypot(xx, yy) * self.distance  # [AU]

        # evaluate the 2D images against the profile
        # to create an image cube
        II = torch.unsqueeze(self.intensity_profile(rr), 0)

        # store deterministic variables for later predictive tests
        # 1D profiles 
        self.iprofile1D = pyro.deterministic("iprofile1D", self.intensity_profile(self.R))

        # 2D images
        self.sky_cube = pyro.deterministic(
            "sky_cube", utils.packed_cube_to_sky_cube(II)
        )

        # convert from Jy/arcsec^2 to Jy/pixel by multiplying by cell_size^2
        self.total_flux = pyro.deterministic(
            "total_flux", self.coords.cell_size**2 * torch.sum(II)
        )

        # packed image with extra channel dimension
        return II
```

We've gone ahead and defined many of our model parameters as latent random variables using `PyroSample`. The prior distribution on these parameters is defined by the `dist...`. For example, with the 

```
self.log_A_0 = PyroSample(dist.Normal(0.0, 0.3))
```
line we've defined the prior on the `log_A_0` parameter to be a Normal distribution with mean $\mu = 0.0$ and standard deviation of $\sigma = 0.3$. 

We have also used multivariate parameters to describe the features of the rings. For example, 

```
self.log_ring_sigmas = PyroSample(
    dist.Normal(0.8, 0.3).expand([self.nrings]).to_event(1)
)
```
has set the prior distribution on each of the (logarithm of the) ring widths to be a Normal distribution with mean of $\mu=0.8$ and standard deviation of $\sigma=0.3$. Not including the central Gaussian envelope, we have 7 rings in this model. The `.expand()` call turns a Normal distribution with a shape of `1` into a distribution with a *batch* shape of 7. This isn't quite what we want in this application, so the `to_event()` call converts the *batch* shape into the *event* shape. For more details on Pyro tensor shapes, we recommend reading the [Tensor shapes in Pyro tutorial](https://pyro.ai/examples/tensor_shapes.html).


When building a new model, we recommend starting out by introducing a set of latent random variables with `PyroSample` and fixing most parameters (by simply defining them as torch tensors, as noted in the comments in the above code). 

### Prior predictive check

Following the advice in [Bayesian Workflow](https://arxiv.org/abs/2011.01808), we'll first test out this model using a *prior predictive check*. This is where we generate random samples from each of the prior distributions and use them to produce versions of the model, in this case, random images of disks with 7 rings. This step is very useful because it helps you identify obvious implementation errors with your model. For example, one design flaw we spotted with an earlier iteration of our code was when we used Normal priors on the ring amplitudes and widths. Both of these values should be positive-valued, which motivated our shift to using Normal priors on the logarithm of the ring amplitudes and widths.

```{code-cell} ipython3
# parameters from Guzman     
distance = 121.0  # pc

# initialize the model 
image_model = PyroDisk(coords=coords, distance=distance)
```

To generate samples from the prior we'll use Pyro's [predictive](https://docs.pyro.ai/en/stable/inference_algos.html#module-pyro.infer.predictive) tool

```{code-cell} ipython3
from pyro.infer import Predictive
```

```{code-cell} ipython3
# initialize a Predictive object, do not condition on any posterior_samples
prior_predictive = Predictive(image_model, num_samples=10)
# call the object to get prior predictive samples
output = prior_predictive()
```

Now let's examine the dictionary of output

```{code-cell} ipython3
output.keys()
```

We see that we now have a dictionary with a list of 10 random samples from the prior. We have the latent random variables that we specified, but we also have the deterministic variables like the 1D profile, total flux, and sky cube. Let's plot up 4 of these sky cubes to get a sense of what we're dealing with.

```{code-cell} ipython3
fig, ax = plt.subplots(nrows=2, ncols=2)

for i, a in enumerate(ax.flatten()):
    a.imshow(output["sky_cube"][i][chan], origin="lower", extent=coords.img_ext, cmap="inferno")
    
plt.tight_layout()
```

And we can visualize the 1D profiles

```{code-cell} ipython3
output["iprofile1D"].shape
```

```{code-cell} ipython3
fig, ax = plt.subplots(nrows=1)

for profile in output["iprofile1D"]:
    ax.plot(image_model.R, profile, color="0.2")
    
ax.set_xlabel("radius [au]")
ax.set_ylabel(r"$I_\nu$ [Jy $\mathrm{arcsec}^{-2}$]");
```

Obviously these do not look exactly like the actual AS 209 disk, and that's OK! These are just samples from the prior distribution; the model hasn't touched any data yet. What is reassuring is that the posterior predictions look like *plausible* disks. For example, they are in roughly the center of the field, there are no negative flux values, inclination and position angle $\Omega$ behave as they should, etc.

Before we move on, though, it would be good to check that we can reproduce a disk that does look like the AS 209 disk using the posterior distributions inferred by Guzmán et al. 2018. To do this we'll use `Predictive` conditioned on a "sample" from the posterior. In reality, we'll just take the maximum a posteriori (MAP) values reported by Guzmán et al. 2018 and treat this as a single sample. Samples are generally reported from the `Predictive` routine as a dictionary of PyTorch tensor arrays, each with length `nsamples`. So we'll need to mimic this structure when providing the Guzmán values to the `posterior_samples` argument.

```{code-cell} ipython3
guzman_values = {'x_centroid': torch.tensor([1.70e-3]),
                'y_centroid': torch.tensor([-3.1e-3]),
                'log_A_0': torch.log10(torch.tensor([1.0])),
                'log_sigma_0': torch.log10(torch.tensor([6.69])), 
                'log_ring_amplitudes': torch.log10(torch.tensor(np.array([[0.274, 0.133, 0.115, 0.074, 0.004, 0.051, 0.008]]))), 
                'ring_means': torch.as_tensor(np.array([[15.13, 27.07, 41.42, 74.08, 91.76, 120.42, 139.06]])), 
                'log_ring_sigmas': torch.log10(torch.tensor(np.array([[7.41, 11.72, 17.40, 7.34, 23.39, 9.84, 23.10]]))),
                 'incl': torch.tensor([34.88 * deg]),
                 'Omega': torch.tensor([85.764 * deg]),
               }

# initialize a Predictive object, condition on the Guzman "posterior sample"
prior_predictive_conditional = Predictive(image_model, posterior_samples=guzman_values, num_samples=1)
output = prior_predictive_conditional()
```

```{code-cell} ipython3
fig, ax = plt.subplots(nrows=1)
ax.imshow(output["sky_cube"][0][chan], origin="lower", extent=coords.img_ext, cmap="inferno");
```

And we see that this looks much more like the AS 209 disk.

+++

### Incorporating the data 

Next, we'll define another class called `VisibilityModel`. This class has an instance of `PyroDisk` as an attribute and takes the image produced by that all the way to the data and evaluates the likelihood function. We could have incorporated all of the functionality inside a single class, but we thought it was cleaner to separate the functionality this way: `PyroDisk` contains the functionality specific to producing images from the Guzmán et al. 2018 model while `VisibilityModel` contains the functionality for producing and evaluating model visibilities.

```{code-cell} ipython3
class VisibilityModel(PyroModule):
    """
    This inherits from the PyroDisk model (which provided Bayesian parameters for the disk model) and extends it to carry the comparison all the way to the data, evaluating a likelihood.

    This will hold the dataset and weights, as well.
    
    The 'device' arg will be used to optionally run our inference on the GPU.
    """

    def __init__(
        self,
        coords=None,
        nchan=1,
        distance=None,
        uu=None,
        vv=None,
        weight=None,
        data=None,
        device=None
    ):
        super().__init__()

        # instantiate the PyroDisk as an attribute to this model
        self.disk = PyroDisk(
            coords=coords,
            nchan=nchan,
            distance=distance,
        )

        # store relevant coords objects
        self.coords = coords
        self.nchan = nchan

        # send the loose data through a DataAverager
        averager = gridding.DataAverager(
            coords=coords,
            uu=uu,
            vv=vv,
            weight=weight,
            data_re=np.real(data),
            data_im=np.imag(data),
        )
        
        self.dataset = averager.to_pytorch_dataset()
        
        # extract relevant quantities
        self.data_re = torch.as_tensor(np.real(self.dataset.vis_indexed).flatten(), device=device)
        self.data_im = torch.as_tensor(np.imag(self.dataset.vis_indexed).flatten(), device=device)
        self.sigma = torch.as_tensor(np.sqrt(1 / self.dataset.weight_indexed).flatten(), device=device)
        
        # objects for forward loop
        self.icube = images.ImageCube(
            coords=self.coords, nchan=self.nchan, passthrough=True
        )

        self.flayer = fourier.FourierCube(coords=coords)

        # create a NuFFT, but only use it for predicting samples
        # store the uu and vv points we might use 
        self.uu = torch.as_tensor(uu, device=device)
        self.vv = torch.as_tensor(vv, device=device)
        self.nufft = fourier.NuFFT(coords=self.coords, nchan=self.nchan)


    def forward(self, predictive=True):
        r"""
        Feed forward to calculate the model visibilities and data likelihood.
        
        Args:
            predictive (boolean): if True, do not condition the model visibilities on the data (generally used when doing posterior predictive checks).
            
        """

        disk_packed_image_cube = self.disk()  # use the PyroDisk to create an ImageCube
        img = self.icube(disk_packed_image_cube)  # identity operation for completeness

        if predictive:
            # use the NuFFT to produce and store samples
            vis_nufft = self.nufft(img, self.uu, self.vv)[0]
            
            pyro.deterministic("vis_real", torch.real(vis_nufft))
            pyro.deterministic("vis_imag", torch.imag(vis_nufft))


        else:
            # evaluate the likelihood
            
            # use the FourierCube layer to get a gridded model
            modelVisibilityCube = self.flayer(img)

            # extract the model visibilities corresponding to the gridded data
            vis = self.dataset(modelVisibilityCube).flatten()

            with pyro.plate("data", len(self.data_re)):
                # condition on the real and imaginaries of the data independently
                pyro.sample(
                    "obs_real", dist.Normal(torch.real(vis), self.sigma), obs=self.data_re
                )
                pyro.sample(
                    "obs_imag", dist.Normal(torch.imag(vis), self.sigma), obs=self.data_im
                )
```

We can also do a prior predictive check with the `VisibilityModel`, just like we did with the `PyroDisk`. The `forward` method of `VisibilityModel` is a bit more complex than a `forward` routine you might find in your average Pyro module. This is because we want to have the best of both worlds when it comes to producing model visibilities and (optionally) evaluating them against data. 

As we described in the [NuFFT](../ci-tutorials/loose-visibilities.md) tutorial, the {class}`mpol.fourier.NuFFT` layer is designed to take an image and produce individual model visibilities corresponding to the $u$ and $v$ sampling locations of the dataset. However, with the large number of visibilities present in your average ALMA dataset ($> 10^5$), computational time can start to be a burden. For many repetitive, computationally heavy tasks like evaluating the likelihood function, we will first grid the visibilities using the {class}`mpol.gridder.DataAverager` and evaluate the likelihood function off of those.

When visualizing model or residual visibility values, it is often far more useful to work with the loose visibility values produced from the NuFFT. This is because the loose visibilities can be gridded using a weighting scheme like Briggs robust weighting, which can dramatically increase the sensitivity of the resulting image. So that is why our `VisibilityModel` uses a {class}`~mpol.fourier.NuFFT` layer to produce model visibilities when working in a predictive mode but otherwise uses a more efficient {class}`~mpol.fourier.FourierCube` layer to produce model visibilities when working in a likelihood evaluation loop.

Now we'll do a predictive check with the `VisibilityModel` using the same disk values found by Guzmán et al. 2018. We will also place it on the GPU with the `.to` call, if the device is available.

```{code-cell} ipython3
# we will use this object throghout the rest of the tutorial, so we'll just call it 'model'
model = VisibilityModel(coords=coords, distance=distance, uu=uu, vv=vv, weight=weight, data=data, device=device)
model.to(device);
```

Because we've added the `PyroDisk` module as an attribute of the `VisibilityModel`, that means that the names of the latent random variables in the `PyroDisk` have changed. We can see that by doing a simple prior predictive check (not conditional)

```{code-cell} ipython3
p_check = Predictive(model, num_samples=1)
output = p_check()
output.keys()
```

This means that we'll need to update the names of some of the parameters in the `guzman_values` dictionary.

```{code-cell} ipython3
guzman_disk_values = guzman_values.copy()
for key in guzman_values:
    guzman_disk_values["disk." + key] = guzman_disk_values.pop(key)
```

```{code-cell} ipython3
guzman_disk_values
```

```{code-cell} ipython3
# initialize a Predictive object, condition on the Guzman "posterior sample"
prior_predictive_conditional_vis = Predictive(model, posterior_samples=guzman_disk_values, num_samples=1)
output = prior_predictive_conditional_vis()
```

We now see that we have `vis_real` and `vis_imag` values in the output samples. These are the "loose" model visibilities produced by the NuFFT layer.

```{code-cell} ipython3
output.keys()
```

To finalize this prior predictive check, we'll grid and image these model and residual visibilities using the same Briggs weighting that we used for the data visibilities. We've written the following function that should help us visualize these quantities, since we'll want to repeat this plot once we've explored the posteriors on our own.

```{code-cell} ipython3
def compare_dirty_model_resid(model_real, model_imag, sky_cube, robust=0.0):

    # convert PyTorch tensors to numpy 
    model_real = model_real.cpu().detach().numpy()
    model_imag = model_imag.cpu().detach().numpy()

    data_real = np.real(data)
    data_imag = np.imag(data)
    
    # calculate the residual visibilities
    resid_real = data_real - model_real 
    resid_imag = data_imag - model_imag

    # use the dirty imager to make images
    img_dirty, _ = make_dirty_image(data_real, data_imag)
    img_model, _ = make_dirty_image(model_real, model_imag)
    img_resid, _ = make_dirty_image(resid_real, resid_imag)
    
    # determine the plot dimensions
    xx = 8 # in
    cax_width = 0.2 # in 
    cax_sep = 0.1 # in
    hmargin = 0.8
    mmargin = 1.2
    lmargin = 0.9
    rmargin = 0.9
    tmargin = 0.3
    bmargin = 0.5
    
    ncol = 2
    nrow = 2
    # the size of image axes + cax_sep + cax_width
    block_width = (xx - lmargin - rmargin - mmargin * (ncol - 1) )/ncol
    ax_width = block_width - cax_width - cax_sep
    ax_height = ax_width 
    yy = bmargin + nrow * ax_height + (nrow - 1) * hmargin + tmargin

    
    fig = plt.figure(figsize=(xx, yy))
        
    ax = []
    cax = []
    for j in range(ncol):
        a = []
        ca = []
        for i in range(nrow):
            a.append(fig.add_axes([(lmargin + i * (block_width + mmargin))/xx, (bmargin + (ax_height + hmargin) * j)/yy, ax_width/xx, ax_height/yy]))
            ca.append(fig.add_axes([(lmargin + i * (block_width + mmargin) + ax_width + cax_sep)/xx, (bmargin + (ax_height + hmargin) * j)/yy, cax_width/xx, ax_height/yy]))
        
        # prepend to list to get order correct
        ax = a + ax
        cax = ca + cax

    cbars = []    
    chan = 0

    comb_img = np.concatenate([img_dirty[chan], img_model[chan]])
    scale_min = np.min(comb_img)
    scale_max = np.max(comb_img)
    
    im_dirty = ax[0].imshow(img_dirty[chan], **kw, vmin=scale_min, vmax=scale_max)
    ax[0].set_title("dirty image")
    cbars.append(plt.colorbar(im_dirty, cax=cax[0]))
    
    im_model = ax[1].imshow(sky_cube.cpu().detach().numpy()[chan], **kw)
    ax[1].set_title("model image")
    cbars.append(plt.colorbar(im_model, cax=cax[1]))
    
    im_model_vis = ax[2].imshow(img_model[chan], **kw, vmin=scale_min, vmax=scale_max)
    ax[2].set_title("model vis imaged")
    cbars.append(plt.colorbar(im_model_vis, cax=cax[2]))
    
    rkw = kw.copy()
    rkw["cmap"] = "bwr_r"
    vvmax = np.max(np.abs(img_resid[chan]))
    im_resid = ax[3].imshow(img_resid[chan], **rkw, vmin=-vvmax, vmax=vvmax)
    ax[3].set_title("residual vis imaged")
    cbars.append(plt.colorbar(im_resid, cax=cax[3]))

    for a in ax:
        a.xaxis.set_major_locator(ticker.MultipleLocator(1))
        a.yaxis.set_major_locator(ticker.MultipleLocator(1))

    for cbar in cbars:
        cbar.set_label(r"Jy/$\mathrm{arcsec}^2$")

    ax[0].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
    ax[0].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")

    for a in ax[1:]:
        a.xaxis.set_ticklabels([])
        a.yaxis.set_ticklabels([])    
        
    
    return fig 
```

```{code-cell} ipython3
fig = compare_dirty_model_resid(output["vis_real"][0], output["vis_imag"][0], output["sky_cube"][0]);
```

Ok, there is still some structure in the residuals, but at least we can be reasonably confident that the Pyro model is producing images that have the right flux and orientation and that the Fourier layers are producing reasonable model visibilities. In the next sections we will do Bayesian inference of the model parameters and hopefully this will deliver us a set that will further reduce the scale of the residuals.

+++

## Parameter inference with Stochastic Variational Inference (SVI)

Now we'll use Stochastic Variational Inference (SVI) to run the inference loop.

```{code-cell} ipython3
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.initialization import init_to_sample

from astropy.io import ascii
from astropy.table import Table
import time
```

```{code-cell} ipython3
model.to(device)

# define SVI guide
guide = AutoNormal(model, init_loc_fn=init_to_sample)

adam = pyro.optim.Adam({"lr": 0.02})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

num_iterations = 15000
pyro.clear_param_store()
loss_tracker = np.empty(num_iterations)

t0 = time.time()
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss_tracker[j] = svi.step(predictive=False)
    
print("Optimization took {:}s".format(time.time() - t0))

# write loss to file 
table = Table()
table["loss"] = np.array(loss_tracker)
ascii.write(table, "loss.csv", overwrite=True)
```

Note that, because we are in a Jupyter notebook tutorial, we don't need to save and then load the output from a run, it's just stored in memory. In a normal workflow, though, you might wish to have one script that runs the optimization loop (perhaps via a batch submission script on a cluster) and then a separate script that plots the results. In that case, you'll want to save the parameter values of the guide after optimization. Here is one way to save them 

```
param_store = pyro.get_param_store()
param_store.save("param_store")

# view items
for key, value in param_store.items():
    print(key, value)
```

And then in your plotting script, you'll want to re-initialize the model and the guide, and then you can load the parameter store into them. For example,

```
# define SVI guide
guide = AutoNormal(model, init_loc_fn=init_to_mean)

param_store = pyro.get_param_store()
param_store.load("param_store")

# need to run the guide step after, otherwise "no stochastic sites"
guide()
```

Now, let's plot the loss values to see how we converged.

```{code-cell} ipython3
table = ascii.read("loss.csv")
# subtract the minimum value 
loss = table["loss"]
loss -= np.min(loss)

# plot loss
fig, ax = plt.subplots(nrows=1)
ax.semilogy(loss)
ax.set_xlabel("iteration")
ax.set_ylabel("loss");
```

### Visualization of samples

We can visualize the posteriors in multiple ways. Since we used an [AutoNormal](https://docs.pyro.ai/en/stable/infer.autoguide.html#autonormal) guide, this means that, by construction, the posteriors will be 1D Gaussians on each parameter, with no covariance between them. (This may be physically unrealistic, which we'll address in a moment). So, one way of reporting the posteriors is simply to report the mean and standard deviation of each of the guide Gaussians. There is a convenience routine, `guide.quantiles()`, that will report the quantiles of the Gaussian distribution for this guide. 

Let's go a step further and examine the posteriors using some visualization routines provided by the [ArviZ](https://python.arviz.org/en/stable/) package. To start, we want to generate samples from the posterior distributions.

As before, we'll use the `Predictive` routine to generate samples. This time, though, we'll pass in the `guide`, which stores the variational distribution that is approximated to the posterior distribution. And, we'll start just by visualizing a subset of the parameters using the `return_sites` argument. 

We can generate samples from the approximate posterior as follows

```{code-cell} ipython3
samples = Predictive(model, guide=guide, return_sites=['disk.incl', 'disk.Omega', 'disk.x_centroid', 'disk.y_centroid', 'disk.log_A_0', 'disk.log_sigma_0', 'disk.log_ring_amplitudes', 'disk.ring_means', 'disk.log_ring_sigmas'], num_samples=2000)(True)

# extract samples from the Pyro Predictive object and convert units for convenience
dict_samples = {k: np.expand_dims(v.detach().numpy(), 0) for k, v in samples.items()}

# convert from radians to degrees
for key in ["disk.incl", "disk.Omega"]:
    dict_samples[key] /= deg
    
# convert from log values
for key in ["disk.log_A_0", "disk.log_sigma_0", "disk.log_ring_amplitudes", "disk.log_ring_sigmas"]:
    new_key = key.replace("log_", "")
    dict_samples[new_key] = 10**dict_samples.pop(key)    
```

and then convert these samples to an ArviZ InferenceData object

```{code-cell} ipython3
import arviz as az
dataset = az.convert_to_inference_data(dict_samples)
dataset
```

Then, it is easy to use the ArviZ plotting routines to make many diagnostic plots. To start, let's visualize the 1D marginal posteriors

```{code-cell} ipython3
az.plot_posterior(dataset, var_names=["disk.Omega", "disk.incl", "disk.A_0", "disk.sigma_0"]);
```

And, we can also visualize the pairwise 2D marginal distributions (often called a "triangle" or "corner" plot)

```{code-cell} ipython3
az.plot_pair(dataset, var_names=["disk.ring_means"]);
```

As we mentioned, the lack of correlation between any parameters is *imposed* by the simple SVI guide that we used. This could be an issue if there were strong correlations between parameters. We'll address this limitiation in the next section by using a guide that incorporates correlations between parameters. 

But first, let's see what the model and residuals look like for this optimized posterior distribution.

```{code-cell} ipython3
samples = Predictive(model, guide=guide, return_sites=['vis_real', 'vis_imag', 'sky_cube'], num_samples=1)(predictive=True)
fig = compare_dirty_model_resid(samples["vis_real"][0], samples["vis_imag"][0], samples["sky_cube"][0]);
```

And the 1D profile -- here we'll overplot 50 draws.

```{code-cell} ipython3
samples = Predictive(model, guide=guide, return_sites=['iprofile1D'], num_samples=50)(predictive=True)

fig, ax = plt.subplots(nrows=1)

for profile in samples["iprofile1D"]:
    ax.plot(model.disk.R, profile, color="k", lw=0.2, alpha=0.2)
    
ax.set_xlabel("radius [au]")
ax.set_ylabel(r"$I_\nu$ [Jy $\mathrm{arcsec}^{-2}$]");
```

We see that there is very little dispersion in these draws from the posterior. This is a feature of the high signal to noise of the dataset but could also be from the parameterization of our model (e.g., not flexible enough, more Gaussian rings required, rings of different shapes, etc...) or the restrictions placed by the `AutoNormal` guide (parameters are uncorrelated). We would expect some of the ring parameters to be correlated with each other (especially those at or below the resolution of the observations), so we'll explore this in the next section.

+++

## SVI with a AutoMultivariateNormal Model

Our first attempt at inference with SVI using the AutoNormal guide seemed to go pretty well. But it's probably unrealistic to assume that there is no correlation between parameters in the model. To address this, we can use a more sophisticated variational guide to approximate the true posterior. 

The next logical step would be to use a guide that still used a Normal distribution to approximate the posterior, but also allowed for correlations between parameters. Fortunately, Pyro provides an `AutoMultivariateNormal` guide that does just this. Let's repeat the SVI process and see what, if anything, changes with our inferred posteriors.

```{code-cell} ipython3
from pyro.infer.autoguide import AutoMultivariateNormal, init_to_mean
```

```{code-cell} ipython3
model.to(device)

# define SVI guide
guide = AutoMultivariateNormal(model, init_loc_fn=init_to_mean)

adam = pyro.optim.Adam({"lr": 0.02})
svi = SVI(model, guide, adam, loss=Trace_ELBO())

num_iterations = 15000
pyro.clear_param_store()
loss_tracker = np.empty(num_iterations)

t0 = time.time()
for j in range(num_iterations):
    # calculate the loss and take a gradient step
    loss_tracker[j] = svi.step(predictive=False)
    
print("Optimization took {:}s".format(time.time() - t0))

# write loss to file 
table = Table()
table["loss"] = np.array(loss_tracker)
ascii.write(table, "loss.csv", overwrite=True)
```

```{code-cell} ipython3
table = ascii.read("loss.csv")
# subtract the minimum value 
loss = table["loss"]
loss -= np.min(loss)

# plot loss
fig, ax = plt.subplots(nrows=1)
ax.semilogy(loss)
ax.set_xlabel("iteration")
ax.set_ylabel("loss");
```

### Visualization of samples

We'll follow a similar procedure as with the `AutoNormal` guide.

```{code-cell} ipython3
samples = Predictive(model, guide=guide, return_sites=['disk.incl', 'disk.Omega', 'disk.x_centroid', 'disk.y_centroid', 'disk.log_A_0', 'disk.log_sigma_0', 'disk.log_ring_amplitudes', 'disk.ring_means', 'disk.log_ring_sigmas'], num_samples=2000)(True)
for k, v in samples.items():
    print(f"{k}: {v.shape}")
```

```{code-cell} ipython3
# extract samples from the Pyro Predictive object and convert units for convenience
dict_samples = {k: np.expand_dims(v.detach().numpy(), 0) for k, v in samples.items()}
# convert from radians to degrees
for key in ["disk.incl", "disk.Omega"]:
    dict_samples[key] /= deg
    
# convert to actual value
for key in ["disk.log_A_0", "disk.log_sigma_0", "disk.log_ring_amplitudes", "disk.log_ring_sigmas"]:
    new_key = key.replace("log_", "")
    dict_samples[new_key] = 10**dict_samples.pop(key)    
    
dataset = az.convert_to_inference_data(dict_samples)
```

Because it is hard to visualize the posteriors for all 27 parameters in a single plot, we will plot pairwise a subset of the variables at a time.

```{code-cell} ipython3
az.plot_pair(dataset, var_names=["disk.ring_means"]);
```

```{code-cell} ipython3
az.plot_pair(dataset, var_names=["disk.ring_sigmas"]);
```

```{code-cell} ipython3
az.plot_pair(dataset, var_names=["disk.ring_amplitudes"]);
```

With the more flexible guide, the correlations between parameters are more accurately captured. Now let's see what the model and residuals look like for this optimized posterior distribution.

```{code-cell} ipython3
samples = Predictive(model, guide=guide, return_sites=['vis_real', 'vis_imag', 'sky_cube'], num_samples=1)(predictive=True)
fig = compare_dirty_model_resid(samples["vis_real"][0], samples["vis_imag"][0], samples["sky_cube"][0]);
```

It's hard to tell much of a difference with the model and residual images.

However, when we plot many draws from the 1D profile

```{code-cell} ipython3
samples = Predictive(model, guide=guide, return_sites=['iprofile1D'], num_samples=50)(predictive=True)

fig, ax = plt.subplots(nrows=1)

for profile in samples["iprofile1D"]:
    ax.plot(model.disk.R, profile, color="k", lw=0.2, alpha=0.2)
    
ax.set_xlabel("radius [au]")
ax.set_ylabel(r"$I_\nu$ [Jy $\mathrm{arcsec}^{-2}$]");
```

We see that there is a *slightly* larger scatter in the draws compared to the `AutoNormal` guide, most noticeable around 40 au. This is because the `AutoMultivariateNormal` guide captured more of the covariance between parameters, resulting in a greater dispersion of draws.

Encouragingly, both our image and 1D profile results compare favorably with those found by [Guzmán et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...869L..48G/abstract) (compare their Figures 2 & 4). 

The true uncertainty in the radial profile may still be underestimated. As we discussed, one source could be the parameterization of the model. In reality, the disk rings are not perfect Gaussian shapes, and so, as currently implemented, our model could never capture the true intensity profile. 


In our opinion, SVI is a very useful inference technique because of its speed and scalability. There is the risk, though, that your guide distribution does not fully capture complex covariances of your posterior distributions. Perhaps some parameter posteriors are significantly non-Gaussian or banana-shaped, and therefore not able to be captured by the multivariate Normal guide. This risk can be hard to assess from SVI fits alone, though there are steps you can take by trying out more [complex guides](https://docs.pyro.ai/en/stable/infer.autoguide.html#) or [writing your own](https://pyro.ai/examples/svi_part_i.html#Guide), parameterized around anticipated covariances.

+++

## Parameter inference with MCMC

If these expanded SVI approaches are unsatisfactory and accurately measuring parameter uncertainties and covariances is critical to your science problem, it may make sense to switch to a more accurate inference algorithm like Markov Chain Monte Carlo (MCMC). With gradient-enabled samplers like Hamiltonian Monte Carlo (HMC) and the No U-Turn Sampler (NUTS), MCMC sampling can still be quite fast compared to traditional MCMC algorithms like Metropolis-Hastings.

To sample this model using MCMC and NUTS, the following steps are required

```
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_sample

model = VisibilityModel(coords=coords, distance=distance, uu=uu, vv=vv, weight=weight, data=data, device=device)
model.to(device)
kernel = NUTS(model, init_strategy=init_to_sample)

mcmc = MCMC(kernel, num_samples=600, warmup_steps=200)
mcmc.run(predictive=False)
samples = mcmc.get_samples()
```

If you will be running this on the GPU (at least as of Pyro 1.8.4), you will also need to change latent variable definitions in `PyroDisk` such that they are instantiated from torch tensors on the GPU, like so

```
self.log_A_0 = PyroSample(dist.Normal(torch.tensor(0.0, device=device), 0.3))
```

This is necessary to place these sample objects on the GPU for use in MCMC (see also this [Pyro issue](https://forum.pyro.ai/t/pyrosample-and-cuda-gpu/4328)) so that you don't get conflicts that some tensors are on the CPU while others are on the GPU. It's not clear to us why this change is necessary for MCMC but not for the SVI algorithms.

Reassuringly, we found that the parameter constraints provided by MCMC were comparable to those provided by SVI with the MultiDiagonal guide. We found that the MCMC NUTS run took about a 1.5 hours to run two independent chains on a GPU. This is still tractable but notably slower than the roughly 5 minutes it took with SVI to find the posterior distributions in this tutorial.

```{code-cell} ipython3

```
