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

# Gridding and diagnostic images

This tutorial provides a brief introduction to complex visibility data, shows how to average it to a "grid", and how to make diagnostic images (e.g., the "dirty image"). The RML imaging workflow is demonstrated in later tutorials.

## Importing data

We'll use a mock CASA measurement set that we produced as part of the [mpoldatasets](https://github.com/MPoL-dev/mpoldatasets) package. One of the tricky things about working with CASA measurement sets is that you need to use CASA to read the visibilities themselves. CASA has historically been packaged as a "monolithic" distribution with its own Python interpreter (which is difficult to install new packages into), but recently, ["modular" CASA](https://casadocs.readthedocs.io/en/stable/notebooks/introduction.html#Modular-Packages) has made it possible to install the CASA routines into your own Python environment (with some restrictions on recent Python version). To avoid introducing a CASA dependence to MPoL, we assume that the user provides the arrays of complex visibilities directly as numpy arrays. The data requirements of RML imaging are really that simple.

In our opinion, the most straightforward way of obtaining the visibilities from a measurement set is to use the CASA ``table`` and ``ms`` tools as described [here](https://mpol-dev.github.io/visread/). The user can use either "monolithic" or "modular" CASA to read the visibilities and save them to a ``.npy`` array on disk. Then, in your normal (less restrictive) Python environment (e.g., Python 3.11) you can read these visibilities and pass them to the MPoL routines.

```{attention}
It's important to remember that MPoL follows the standard baseline convention as laid out in [Thompson, Moran, and Swenson](https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract) and other radio interferometry textbooks, while CASA follows a [historically complicated convention](https://casa.nrao.edu/casadocs/casa-5.5.0/knowledgebase-and-memos/casa-memos/casa_memo2_coordconvention_rau.pdf/view) derived from AIPS. The difference between the two can be expressed as the complex conjugate of the visibilities. So, if you find that your image appears upside down and mirrored, you'll want to take ``np.conj`` of your visibilities.
```

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.data import download_file
```

```{code-cell}
# load the mock dataset of the ALMA logo
fname = download_file(
    "https://zenodo.org/record/4930016/files/logo_cube.noise.npz",
    cache=True,
    show_progress=True,
    pkgname="mpol",
)
```

```{code-cell}
d = np.load(fname)
uu = d["uu"]
vv = d["vv"]
weight = d["weight"]
data = d["data"]
data_re = np.real(data)
data_im = np.imag(data)
```

## Plotting the data

Following some of the exercises in the [visread documentation](https://mpol-dev.github.io/visread/tutorials/introduction_to_casatools.html), let's plot up the baseline distribution and get a rough look at the raw visibilities. For more information on these data types, we recommend you read the [Introduction to RML Imaging](../rml_intro.md).

Note that the `uu`, `vv`, `weight`, `data_re`, and `data_im` arrays are all two-dimensional numpy arrays of shape `(nchan, nvis)`. This is because MPoL has the capacity to image spectral line observations. MPoL will absolutely still work with single-channel continuum data, you will just need to work with 2D arrays of shape `(1, nvis)`.

For this particular dataset,

```{code-cell}
nchan, nvis = uu.shape
print("Dataset has {:} channels".format(nchan))
print("Dataset has {:} visibilities".format(nvis))
```

Therefore, understand that the following baseline and visibility scatter plots are showing about a third of a million points.

Here, we'll plot the baselines corresponding to the first channel of the dataset by simply marking a point for every spatial frequency coordinate, $u$ and $v$, in the dataset

```{code-cell}
fig, ax = plt.subplots(nrows=1)
ax.scatter(uu[0], vv[0], s=1.5, rasterized=True, linewidths=0.0, c="k")
ax.set_xlabel(r"$u$ [k$\lambda$]")
ax.set_ylabel(r"$v$ [k$\lambda$]");
```

The fact that visibility data has two spatial frequency coordinates sometimes makes plotting representations of the data values a little challenging. To simplify things, let's define a 1D "radial" visibility coordinate as $q = \sqrt{u^2 + v^2}$, and plot the real, imaginary, amplitude, and phase values of the visibilities against this.


```{code-cell}
qq = np.hypot(uu, vv)

amp = np.abs(data)
phase = np.angle(data)

chan = 0

pkw = {"s":1.5, "rasterized":True, "linewidths":0.0, "c":"k"}

fig, ax = plt.subplots(nrows=4, sharex=True)

ax[0].scatter(qq[chan], data[chan].real, **pkw)
ax[0].set_ylabel("Re(V) [Jy]")

ax[1].scatter(qq[chan], data[chan].imag, **pkw)
ax[1].set_ylabel("Im(V) [Jy]")

ax[2].scatter(qq[chan], amp[chan], **pkw)
ax[2].set_ylabel("amplitude [Jy]")

ax[3].scatter(qq[chan], phase[chan], **pkw)
ax[3].set_ylabel("phase [radians]")
ax[3].set_xlabel(r"$q$ [k$\lambda$]");
```

There are nearly a third of a million points in each figure, and each is quite noisy, so we can't learn much from this plot alone. But we should be reassured that we see similar types of scatter as we might observe were we to inspect the raw data using CASA's [plotms](https://casadocs.readthedocs.io/en/v6.5.2/api/tt/casaplotms.plotms.html?highlight=plotms) tool.


## The {class}`~mpol.coordinates.GridCoords` object

Now, lets familiarize ourselves with MPoL's {class}`mpol.coordinates.GridCoords` object.

```{code-cell}
from mpol import coordinates, gridding
```

Two numbers, `cell_size` and `npix`, uniquely define a grid in image space and in Fourier space.

```{code-cell}
coords = coordinates.GridCoords(cell_size=0.005, npix=800)
```

The {class}`mpol.coordinates.GridCoords` object is mainly a container for all of the information about this grid. You can see all of the properties accessible in the {py:class}`mpol.coordinates.GridCoords` API documentation. The information you'll most likely want to access are the image dimensions

```{code-cell}
coords.img_ext  # [arcsec]
```

which are meant to feed into the `extent` parameter of `matplotlib.pyplot.imshow`.

+++

## Making images with {class}`~mpol.gridding.DirtyImager`

Those familiar with radio astronomy will be familiar with the idea of "gridding" loose visibilities to a Cartesian $u,v$ grid. MPoL has two classes that "grid" visibilities: {class}`mpol.gridding.DirtyImager` and {class}`mpol.gridding.DataAverager`. Their internals may be similar, but they serve different purposes. First, let's look at how we can use the {class}`mpol.gridding.DirtyImager` to make diagnostic images using the inverse Fast Fourier Transform, frequently called the "dirty image" by radio astronomers.

We can instantiate a {class}`~mpol.gridding.DirtyImager` object by

```{code-cell}
imager = gridding.DirtyImager(
    coords=coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data_re,
    data_im=data_im,
)
```

Instantiating the {class}`~mpol.gridding.DirtyImager` object attaches the {class}`~mpol.coordinates.GridCoords`  object and the loose visibilities. There is also a convenience method to create the {class}`~mpol.coordinates.GridCoords` and {class}`~mpol.gridding.DirtyImager` object in one shot by

```{code-cell}
imager = gridding.DirtyImager.from_image_properties(
    cell_size=0.005,  # [arcsec]
    npix=800,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data_re,
    data_im=data_im,
)
```

if you don't want to specify your {class}`~mpol.coordinates.GridCoords` object separately.

As we saw, the raw visibility dataset is a set of complex-valued Fourier samples. Our objective is to make images of the sky-brightness distribution and do astrophysics. We'll cover how to do this with MPoL and RML techniques in later tutorials, but it is possible to get a rough idea of the sky brightness by calculating the inverse Fourier transform of the visibility values.

To do this, you can call the {meth}`mpol.gridding.DirtyImager.get_dirty_image` method on your {class}`~mpol.gridding.DirtyImager` object. This routine will average, or 'grid', the loose visibilities to the Fourier grid defined by {class}`~mpol.coordinates.GridCoords` and then calculate the diagnostic dirty image and dirty beam cubes that correspond to the Fourier transform of the gridded visibilities.

There are several different schemes by which to do the averaging, each of which will deliver different image plane resolutions (defined by the size of the PSF or dirty beam) and thermal noise properties. MPoL implements 'uniform', 'natural', and 'briggs' robust weighting. For more information on the difference between these schemes, see the [CASA documentation](https://casa.nrao.edu/casadocs-devel/stable/imaging/synthesis-imaging/data-weighting) or Chapter 3 of Daniel Briggs' [Ph.D. thesis](http://www.aoc.nrao.edu/dissertations/dbriggs/).

```{code-cell}
img, beam = imager.get_dirty_image(weighting="briggs", robust=0.0)
```

Note that these are three dimensional image cubes with the same `nchan` as the input visibility data.

```{code-cell}
print(beam.shape)
print(img.shape)
```

And the image has 'units' of "Jy/beam". 

```{margin} Dirty Image Units
N.B. that the intensity units of the dirty image are technically undefined. The fact that MPoL assigns units of "Jy/beam" here is truly a stop-gap measure meant to provide a rough diagnostic check that the data has been correctly imported and to enable comparison to a CASA dirty image, for example. For more details on the ill-defined units of the dirty image (i.e., "Jy/dirty beam"), see footnote 4 of Chapter 3.2 of Daniel Briggs' [Ph.D. thesis](http://www.aoc.nrao.edu/dissertations/dbriggs/) or the discussion in [Czekala et al. 2021b](https://ui.adsabs.harvard.edu/abs/2021ApJS..257....2C/abstract).
```

```{code-cell}
chan = 4
kw = {"origin": "lower", "interpolation": "none", "extent": imager.coords.img_ext}
fig, ax = plt.subplots(ncols=2, figsize=(6.0, 4))
ax[0].imshow(beam[chan], **kw)
ax[0].set_title("beam")
ax[1].imshow(img[chan], **kw)
ax[1].set_title("image")
for a in ax:
    a.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
    a.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
fig.subplots_adjust(left=0.14, right=0.90, wspace=0.35, bottom=0.15, top=0.9)
```

If you were working with this measurement set in CASA, it's a good idea to compare the dirty image produced here to the dirty image from CASA (i.e., produced by `tclean` with zero CLEAN iterations). You should confirm that these two dirty images look very similar (i.e., nearly but most likely not quite to numerical precision) before moving on to regularized maximum imaging. If your image appears upside down or mirrored, check whether you converted your visibility data from the CASA baseline convention to the regular TMS baseline convention by complex-conjugating your visibilities.


## Averaging and exporting data with {class}`~mpol.gridding.DataAverager`

As we saw at the beginning of this tutorial, an ALMA dataset may easily contain 1/3 million or more individual visibility measurements, which can present a computational burden for some imaging routines. Just like many noisy data points can be "binned" into a set of fewer higher signal to noise points (for example, as with a lightcurve of a transiting exoplanet), so too can visibility data points be averaged down. 

To do this, you can instantiate a {class}`~mpol.gridding.DataAverager` object and then call the {meth}`mpol.gridding.DataAverager.to_pytorch_dataset` method. This routine will average, or 'grid', the loose visibilities to the Fourier grid defined by {class}`~mpol.coordinates.GridCoords` and then export the dataset as a {class}`mpol.datasets.GriddedDataset` object.

```{code-cell}
averager = gridding.DataAverager(
    coords=coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data_re,
    data_im=data_im,
)

dset = averager.to_pytorch_dataset()
```


## Checking data weights
When working with real data, it is possible that the statistical uncertainties---conveyed by the weights---were [not correctly calibrated by certain CASA versions](https://mpol-dev.github.io/visread/tutorials/rescale_AS209_weights.html). For dirty and CLEAN imaging purposes, it's OK if the weights are not correctly scaled so long as their *relative* scalings are correct (to each other). For forward-modeling and RML imaging, it's important that the weights are correctly scaled in an absolute sense. To alert the user to the possibility that their weights may be incorrectly calibrated, the routines internal to {class}`~mpol.gridding.DirtyImager` will raise a ``RuntimeWarning`` if the weights are incorrectly scaled. Even though the weights are incorrect, the user image may still want the dirty image---hence why these routines issue a warning instead of an error.

```
  img, beam = imager.get_dirty_image(
        weighting="uniform", check_visibility_scatter=True, max_scatter=1.2
  )
```

However, if the user goes to export the gridded visibilities as a PyTorch dataset for RML imaging using {class}`~mpol.gridding.DataAverager`, incorrectly scaled weights will raise a ``RuntimeError``. RML images and forward modeling inferences will be compromised if the weights are not statistically valid.

The sensitivity of the export routines can be adjusted by changing the ``max_scatter`` keyword. Scatter checking can be disabled by setting ``check_visibility_scatter=False``, but is not recommended unless you are trying to debug things.

```
  dset = averager.to_pytorch_dataset(check_visibility_scatter=True, max_scatter=1.2)
```
