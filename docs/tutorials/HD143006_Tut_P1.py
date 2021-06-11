# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + nbsphinx="hidden"
# %matplotlib inline

# + nbsphinx="hidden"
# %run notebook_setup
# -
# # HD143006 Tutorial Part 1
# ### Introduction to Regularized Maximum Likelihood (RML) Imaging
# 
# 
# Regularized Maximum Likelihood (RML) imaging is a forward modeling methodology that is growing in popularity in Very Long Baseline Interferometry. Unlike the inverse modeling done with the CLEAN algorithm and self-calibration, RML represents our predicted image as an array of pixels, performs a forward Fourier transform on this array to the visibility domain and assesses the possibility of this image by comparing the visibility domain of the predicted image to the measured data. When collecting data with a limited number of interferometers much of the information is lost. This means there are several predicted images that would have agreement with the true visibilities even though the predicted image may not be correct in the portions of the image that were lost. This is where "Maximum Likelihood" in RML is represented. Of the several images that are possible representations, we know some are more likely than others- it is unlikely that we will see a mermaid shaped accretion disk and more reasonable to see a disk shape! In addition to knowing how likely our prediction is, we also need to know how good it is based on criterion we specify. This is the role of the regularizer. The regularizer can be set to different favoring criterion (e.g. smoothness across pixels would facilitate use of a TV regularizer), this will help fill in the missing data and assist in determining what pixels we can keep or adjust in our next prediction. This assesment of the predicted image is contained in an objective _or loss_ function that we optimize by changing pixel parameters until we reach a predicted image that has minimum loss thus giving us the best representation of the image we can provide. 
# 
# 
# ### Loading Data
# Let's load the data as we've done in previous tutorials. We will be examining the Fiducial Images (.fits) and Extracted Visibilities (.npz) of HD143006 DSHARP Object. The extracted visibilities were calculated from the Final Calibrated Measurement Sets from DSHARP [here](https://mpol-dev.github.io/visread/) using visread.
# 
# *You can either download these two files (HD143006_continuum.fits and HD143006_continuum.npz) directly to your working directory, or use astropy to download them during run time.*
# 
#

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.data import download_file

# downloading fits file
fname_F = download_file(
   'https://almascience.nrao.edu/almadata/lp/DSHARP/images/HD143006_continuum.fits',
   cache=True,
   pkgname='mpol',
)

# downloading extracted visibilities file
fname_EV = download_file(
    'https://zenodo.org/record/4904794/files/HD143006_continuum.npz',
    cache=True,
    pkgname='mpol',
)

# Now that we have the files, let us examine the fits image created by the DSHARP team using the CLEAN algorithm. 

# opening the fits file
dfits = fits.open(fname_F)
# printing out the header info
dfits.info()
# getting the data from the fits file and closing the fits file
clean_fits = dfits[0].data
dfits.close()
# plotting the clean fits
plt.imshow(np.squeeze(clean_fits), origin='lower')
#limit the x and y axes so that we only plot the disk
plt.xlim(left=1250, right=1750)
plt.ylim(top=1750, bottom=1250)

# This is the image produced by the CLEAN algorithm used by the DSHARP team. We will compare this with the image produced using RML through MPoL. While in this tutorial we will only be creating an MPoL Gridder object and plotting the dirty image, **Part 2** of the tutorial will cover the optimization loop of the model for image quality improvements. 
# 
# To create the dirty image, we will use the extracted visibilities from the npz file and the MPoL Gridder and Coordinates packages.
# 

# load extracted visibilities from npz file
dnpz = np.load(fname_EV) 
uu = dnpz['uu']
vv = dnpz['vv']
weight = dnpz['weight']
data = dnpz['data']

# Let's quickly plot the U,V visibilities as seen in the [Cross Validation Tutorial](https://mpol-dev.github.io/MPoL/tutorials/crossvalidation.html) and the [Visread docs](https://mpol-dev.github.io/visread/tutorials/introduction_to_casatools.html#Get-the-baselines).

fig, ax = plt.subplots(nrows=1)
ax.scatter(uu,vv, s=.5, rasterized=True, linewidths=0.0, c='k')
ax.scatter(-uu,-vv, s=.5, rasterized=True, linewidths=0.0, c='k')
ax.set_xlabel(r"$u$ [k$\lambda$]")
ax.set_ylabel(r"$v$ [k$\lambda$]")
ax.set_title(r'$U$, $V$ Visibilities')

# As you can see, there are very few visibilites > 7,000 ($k\lambda$), and a very dense region of visibilities between -2000 and 2000 ($k\lambda$). This indicates outliers at the higher frequencies while the bulk of our data stems from these lower frequencies.

# To create the MPoL Gridder object, we need a `cell_size` and the number of pixels in the width of our image, `npix`. You can read more about these properties in the [GridCoords] (https://mpol-dev.github.io/MPoL/api.html#mpol.coordinates.GridCoords) API Documentation.  In the fits header we see our image is 3000x3000 pixels, so `npix=3000`. Getting our cell size in terms of arcseconds is a bit more tricky. Our fits image has a header called `CDELT1` which is the scaling in degrees. To get this into arcseconds we multiply by 3600. We save this as `cdelt_scaling`. `cdelt_scaling` can be negative, and cell_size must be positive so we will take the absolute value of this.

# opening the fits file
dfits = fits.open(fname_F)
cdelt_scaling = dfits[0].header['CDELT1'] * 3600 # scaling [arcsec]
cell_size = abs(cdelt_scaling) # [arcsec]
#close fits file
dfits.close()

from mpol import gridding, coordinates

# creating Gridder object
coords = coordinates.GridCoords(cell_size=cell_size, npix=3000)
gridder = gridding.Gridder(
    coords = coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data.real, # seperating the real and imaginary values of our data
    data_im=data.imag
)

# We now have everything we need to get the MPoL dirty image. Here we are using [Gridder.get_dirty_image()](../api.rst#mpol.gridding.Gridder.get_dirty_image) to average the visibilities to the grid defined by gridder and from there we get our dirty image and dirty beam. There are different ways to average the visibilities, called weighting, and here we use Uniform and Briggs weighting to find and produce a dirty image that resembles the CLEAN image. More info on the weighting can be read in the [CASA documentation](https://casa.nrao.edu/casadocs-devel/stable/imaging/synthesis-imaging/data-weighting). For the Briggs weighting, we will use three different values for the `robust` variable. This dictates how aggresive our weight scaling is towards image resolution or image sensitivity. 
# 
# *Note: when `robust=-2.0` the result is similar to that of the Uniform scale*

img, beam = gridder.get_dirty_image(weighting='uniform')
img1, beam1 = gridder.get_dirty_image(weighting="briggs", robust=1.0, unit="Jy/arcsec^2")
img2, beam2 = gridder.get_dirty_image(weighting="briggs", robust=0.0, unit="Jy/arcsec^2")
img3, beam3 = gridder.get_dirty_image(weighting="briggs", robust=-1.0, unit="Jy/arcsec^2")

# Great! Now let's make a plotting function to show us the MPoL image. If you have read through other MPoL tutorials, then this code should look familiar. We are going to plot all four of the different weightings, so creating a plotting function simplifies our code a lot.

def plot(img, savename="image.png", imtitle="image"):
    kw = {"origin": "lower", "extent": gridder.coords.img_ext}
    fig, ax = plt.subplots(ncols=1)
    im = ax.imshow(np.squeeze(img), **kw)
    plt.colorbar(im)
    ax.set_title(imtitle)
    ax.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
    ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
    fig.savefig(savename, dpi=300)
    plt.xlim(left=-.75, right=.75)
    plt.ylim(bottom=-.75, top=.75)
    return ax

plot(img, savename="uniform.png", imtitle="uniform")
plot(img1, savename="robust_1.0.png", imtitle="robust_1.0")
plot(img2, savename="robust_0.png", imtitle="robust_0")
plot(img3, savename="robust_-1.0.png", imtitle="robust_-1.0")

# Below we plot the DSHARP CLEAN image alongside the MPoL RML Dirty Image weighted with the Briggs scale and `robust=0.0` for comparison.

kw = {"origin": "lower", "extent": gridder.coords.img_ext}
fig, ax = plt.subplots(ncols = 2)
ax[0].imshow(np.squeeze(clean_fits), **kw)
ax[0].set_title('DSHARP CLEAN Image')
ax[0].set_xlim(left=-.75, right=.75)
ax[0].set_ylim(bottom=-.75, top=.75)
ax[0].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[0].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
ax[1].imshow(np.squeeze(img2), **kw)
ax[1].set_title('MPoL RML Dirty Image')
ax[1].set_xlim(left=-.75, right=.75)
ax[1].set_ylim(bottom=-.75, top=.75)
ax[1].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[1].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
plt.tight_layout()


# As you can see, the Dirty Image created by MPoL looks very similar to the image the DSHARP team came up with using the CLEAN algorithm. While the RML Dirty Image appears more noisy, it also appears sharper and features are more distinct. Using a very simple process with very little data processing MPoL is able to produce an image of comparable quality to the final product of the CLEAN algorithm. In the next part of the HD143006 tutorial, we will be cleaning the image and explaining how we can get an even better result using Neural Networks, Optimization, and Cross Validation with help from [PyTorch](pytorch.org). 
# 
