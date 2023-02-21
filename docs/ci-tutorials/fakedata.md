---
jupytext:
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

%run notebook_setup
```

(mock-dataset-label)=
# Making a Mock Dataset

In this tutorial, we'll explore how you might construct a mock dataset from a known sky brightness distribution. In many ways, this problem is already solved in a realistic manner by CASA's [simobserve](https://casadocs.readthedocs.io/en/latest/api/tt/casatasks.simulation.simobserve.html) task. However, by replicating the key parts of this process with MPoL framework, we can easily make direct comparisons to images produced using RML techniques.

In a nutshell, this process is works by
1. taking a known sky brightness distribution (i.e., a mock "true" image)
2. inserting it into an {class}`mpol.images.ImageCube`
3. using the {class}`mpol.fourier.NuFFT` to predict visibilities at provided $u,v$ locations
4. adding noise

The final two steps are relatively straightforward. The first two steps are conceptually simple but there are several technical caveats one should be aware of, which we'll cover now.

## Choosing a mock sky brightness distribution

You can choose a mock sky brightness distribution from a simulation, a parametric model, or even an image from the Internet. For this tutorial, we'll use a JPEG image from the internet, since it will highlight many of the problems one might run into. First, we'll download the image and display it.

```{code-cell} ipython3
# use python to download an image
import requests

image_url="https://cdn.eso.org/images/screen/alma-eight_close.jpg"

img_data = requests.get(image_url).content
with open('alma.jpg', 'wb') as handler:
    handler.write(img_data)
```

```{code-cell} ipython3
---
mystnb:
  figure:
    caption: 'The ALMA antennas. Credit: ALMA (ESO/NAOJ/NRAO)'
    name: alma-ref
  image:
    alt: ALMA
    classes: shadow bg-primary
    width: 600px
---
from IPython.display import Image
Image("alma.jpg")
```

There are several operations we will need to perform on this image before it is suitable to insert into an {class}`mpol.images.ImageCube`.
1. convert the JPEG image to greyscale `float64` values
1. make the image square (if necessary)
1. choose a target {class}`~mpol.images.ImageCube` size via `cell_size` and `npix`. The range of acceptable image dimensions depends on the range $u,v$ coordinates
1. if the raw image size is larger than the target size of the {class}`~mpol.images.ImageCube`, smooth and downsample the image
1. scale flux units to be Jy/arcsec^2

To perform image manipulations, we'll use the [Pillow](https://pillow.readthedocs.io/en/stable/index.html) library.

## Using Pillow to greyscale, apodize, pad, and resize

```{code-cell} ipython3
from PIL import Image, ImageOps, ImageMath
import numpy as np

im_raw = Image.open("alma.jpg")

# convert to greyscale
im_grey = ImageOps.grayscale(im_raw)

# get image dimensions
xsize, ysize = im_grey.size
print(im_grey.mode)
```

In converting the JPEG image to greyscale (mode "L" or "luminance"), the Pillow library has reduced the color channel to a single axis with an 8-bit unsigned integer, which can take on the values from 0 to 255. More info on the modes is available [here](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes). We can see the greyscale image

```{code-cell} ipython3
im_grey
```

Now let's think about how to make the image square. Depending on the image, we can either crop the longer dimension or pad the larger dimension. Before we do that, though, we also need to think about the image edges.

Because the discrete Fourier transform is used to take an image to the visibility plane, we make the assumption that the image is infinite and periodic in space beyond the field of view. i.e., it tiles to infinity. Therefore, to avoid introducing spurious spatial frequencies from discontinous edges, it is a good idea to make sure that the edges of the image all have the same value. The simplest thing to do here is to taper the image edges such that they all go to 0. We can do this by multiplying by the image by an apodization function, like the Hann window. We'll multiply two 1D Hann windows to create a 2D apodization window.

```{code-cell} ipython3
xhann = np.hanning(xsize)
yhann = np.hanning(ysize)
# each is already normalized to a max of 1
# so hann is also normed to a max of 1
# broadcast to 2D
hann = np.outer(yhann, xhann)

# now convert the numpy array to a Pillow object
# scale to 0 - 255 and then convert to uint8
hann8 = np.uint8(hann * 255)
im_apod = Image.fromarray(hann8)
```

We can visualize the 2D Hann apodization window

```{code-cell} ipython3
im_apod
```

And then use image math to multiply the apodization window against the greyscaled image

```{code-cell} ipython3
im_res = ImageMath.eval("a * b", a=im_grey, b=im_apod)
```

To give an image with a vignette-like appearance.

```{code-cell} ipython3
im_res
```

Now, let's pad the image to be square

```{code-cell} ipython3
max_dim = np.maximum(xsize, ysize)
im_pad = ImageOps.pad(im_res, (max_dim, max_dim))
```

```{code-cell} ipython3
im_pad
```

Great, we now have a square, apodized image.
```{margin} Simulations
We should note that all of these pre-processing steps were only necessary because we pulled a non-square JPEG image from the internet. If we were starting from an image produced from a radiative transfer situation (for example, a solitary protoplanetary disk in the center of a field), we could skip most of these previous steps.
```
The next thing we should fix is that a 1280 x 1280 image is still a bit too many pixels for most ALMA observations. I.e., the spatial resolution or "beam size" of most ALMA observations is such that for any single-pointing observation, we wouldn't need this many pixels to represent the full information content of the image. Therefore, let's resize the image to be a bit smaller.

```{code-cell} ipython3
npix = 500
im_small = im_pad.resize((npix,npix))
```

```{code-cell} ipython3
im_small
```

## Exporting to a Numpy array and setting flux scale

Now that we have done the necessary image preparation, we're ready to leave the Pillow library and work with numpy arrays and pytorch tensors. First we convert from a Pillow object to a numpy array

```{code-cell} ipython3
import numpy as np
a = np.array(im_small)
```

We can see that this array is now a 32-bit integer array (it was promoted from an 8-bit integer array during the ImageMath operation to save precision).

```{code-cell} ipython3
a
```

We will convert this array to a `float64` type and normalize its max value to 1.

```{code-cell} ipython3
b = a.astype("float64")
b = b/b.max()
```

Now, we can plot this array using matplotlib's `imshow` and using the `origin="lower"` argument as we might normally do with arrays of data from MPoL.
```{margin} MPoL Image Orientations
Now might be a good time to familiarize yourself with the {ref}`cube-orientation-label`, if you aren't already familiar.
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.imshow(b, origin="lower")
```

In doing so, we've uncovered an additional problem that the image is upside down! We can fix this using

```{code-cell} ipython3
c = np.flipud(b)
plt.imshow(c, origin="lower")
```

In this example, we're only working with a single-channel mock sky brightness distribution, so we only need to add an extra channel dimension to make an image cube. If we were working with a multi-channel sky brightness distribution, we could repeat the above transformations for each channel of the image cube.

```{code-cell} ipython3
d = np.expand_dims(c, axis=0)
```

Now let's choose how big we want our mock sky brightness to be on the sky. Adjusting the `cell_size` changes the maximum spatial frequency that can be represented in the image. I.e., a smaller pixel `cell_size` will enable an image to carry higher spatial frequencies. Changing the number of pixels in the image via `npix` will change the number of $u,v$ cells between 0 and the max spatial frequency. We effectively chose the `npix` when we performed the resize operation, so all that's left is to choose the `cell_size`.

```{code-cell} ipython3
cell_size = 0.03 # arcsec
```

The final task is to scale the amplitude of the image to the desired level. The {class}`~mpol.images.ImageCube` object will expect the input tensor to be in units of Jy/arcsec^2.

Let's assume that we would like the total flux of our mock image to be 30 Jy, which a very bright source for ALMA band 6. Then again, the noise levels in the mock baseline distribution we plan to use are relatively high, the baseline distribution lacks short spacings, and we want to make sure our source shines through.

So, if we have assigned each pixel to be 0.03 arcseconds on each side, then each pixel has an area of

```{code-cell} ipython3
pixel_area = cell_size**2 # arcsec
print(pixel_area, "arcsec^2")
```

What is the current flux of the image?

```{code-cell} ipython3
# if the raw image is supposed to be in Jy/arcsec^2, then to calculate
# total flux, we would convert to Jy/pixel by multiplying area / pixel
# and then summing all values
old_flux = np.sum(d * pixel_area)
print(old_flux, "Jy")
```

So, if we want the image to have a total flux of 30 Jy, we need to multiply by a factor of

```{code-cell} ipython3
flux_scaled = 30/old_flux * d
```

```{code-cell} ipython3
print("Total flux of image is now {:.1f} Jy".format(np.sum(flux_scaled * pixel_area)))
```

## Initializing {class}`~mpol.images.ImageCube`

Now, we'll convert the numpy array to a PyTorch tensor

```{code-cell} ipython3
import torch
img_tensor = torch.tensor(flux_scaled.copy())
```

And finally, we'll shift the tensor from a "Sky Cube" to a "Packed Cube" as the {class}`~mpol.images.ImageCube` expects

```{code-cell} ipython3
from mpol import utils
img_tensor_packed = utils.sky_cube_to_packed_cube(img_tensor)
```

```{code-cell} ipython3
from mpol.images import ImageCube
image = ImageCube.from_image_properties(cell_size=cell_size, npix=npix, nchan=1, cube=img_tensor_packed)
```

If you want to double-check that the image was correctly inserted, you can do
```
# double check it went in correctly
plt.imshow(np.squeeze(utils.packed_cube_to_sky_cube(image()).detach().numpy()), origin="lower")
```
to see that it's upright and not flipped.

## Obtaining $u,v$ baseline and weight distributions

One of the key use cases for producing a mock dataset from a known sky brightness is to test the ability of an RML algorithm to recover the "true" image. $u,v$ baseline distributions from real interferometric arrays like ALMA, VLA, and others are highly structured sampling distributions that are difficult to accurately replicate using distributions available to random number generators.

Therefore, we always recommend generating fake data using $u,v$ distributions from real datasets, or use those produced using realistic simulators like CASA's [simobserve](https://casadocs.readthedocs.io/en/latest/api/tt/casatasks.simulation.simobserve.html) task. In this example, we'll just use the baseline distribution from the mock dataset we've used in many of the tutorials. You can see a plot of it in the [Gridding and Diagnostic Images](gridder.md) tutorial. We'll only need the $u,v$ and weight arrays.

```{code-cell} ipython3
from astropy.utils.data import download_file

# load the mock dataset of the ALMA logo
fname = download_file(
    "https://zenodo.org/record/4930016/files/logo_cube.noise.npz",
    cache=True,
    show_progress=True,
    pkgname="mpol",
)

# select the components for a single channel
chan = 4
d = np.load(fname)
uu = d["uu"][chan]
vv = d["vv"][chan]
weight = d["weight"][chan]
```

MPoL has a helper routine to calculate the maximum `cell_size` that can still Nyquist sample the highest spatial frequency in the baseline distribution.

```{code-cell} ipython3
max_uv = np.max(np.array([uu,vv]))
max_cell_size = utils.get_maximum_cell_size(max_uv)
print("The maximum cell_size that will still Nyquist sample the spatial frequency represented by the maximum u,v value is {:.2f} arcseconds".format(max_cell_size))
assert cell_size < max_cell_size
```

Thankfully, we see that we already chose a sufficiently small `cell_size`.

## Making the mock dataset

With the {class}`~mpol.images.ImageCube`, $u,v$ and weight distributions now in hand, generating the mock visibilities is relatively straightforward using the {func}`mpol.fourier.make_fake_data` routine. This routine uses the {class}`~mpol.fourier.NuFFT` to produce loose visibilities at the $u,v$ locations and then adds random Gaussian noise to the visibilities, drawn from a probability distribution set by the value of the weights.

```{code-cell} ipython3
from mpol import fourier
# will have the same shape as the uu, vv, and weight inputs
data_noise, data_noiseless = fourier.make_fake_data(image, uu, vv, weight)

print(data_noise.shape)
print(data_noiseless.shape)
print(data_noise)
```

Now you could save this to disk. Since this is continuum dataset, we'll remove the channel dimension from the mock visibilities

```{code-cell} ipython3
data = np.squeeze(data_noise)
# data = np.squeeze(data_noiseless)
np.savez("mock_data.npz", uu=uu, vv=vv, weight=weight, data=data)
```

And now you could use this dataset just like any other when doing RML inference, and now you will have a reference image to compare "ground truth" to.

+++

## Verifying the mock dataset

To make sure the whole process worked OK, we'll load the visibilities and then make a dirty image. We'll set the coordinates of the gridder and dirty image to be exactly those as our input image, so that we can make a pixel-to-pixel comparison. Note that this isn't strictly necessary, though. We could make a range of images with different `cell_size`s and `npix`s.

```{code-cell} ipython3
from mpol import coordinates, gridding

# well set the
coords = coordinates.GridCoords(cell_size=cell_size, npix=npix)

imager = gridding.DirtyImager(
    coords=coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=np.squeeze(np.real(data)),
    data_im=np.squeeze(np.imag(data)),
)
```

```{code-cell} ipython3
C = 1 / np.sum(weight)
noise_estimate = C * np.sqrt(np.sum(weight))
print(noise_estimate, "Jy / dirty beam")
```

```{code-cell} ipython3
img, beam = imager.get_dirty_image(weighting="briggs", robust=1.0, unit="Jy/arcsec^2")
```

```{code-cell} ipython3
chan = 0
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

We can even subtract this on a pixel-by-pixel basis and compare to the original image.

```{code-cell} ipython3
chan = 0
kw = {"origin": "lower", "interpolation": "none", "extent": imager.coords.img_ext}
fig, ax = plt.subplots(ncols=3, figsize=(6.0, 3))

ax[0].imshow(flux_scaled[chan], **kw)
ax[0].set_title("original")

ax[1].imshow(img[chan], **kw)
ax[1].set_title("dirty image")

ax[2].imshow(flux_scaled[chan] - img[chan], **kw)
ax[2].set_title("difference")

ax[0].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[0].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")

for a in ax[1:]:
    a.xaxis.set_ticklabels([])
    a.yaxis.set_ticklabels([])

fig.subplots_adjust(left=0.14, right=0.90, wspace=0.2, bottom=0.15, top=0.9)
```

The subtraction revears some interesting artefacts.
1. the dirty image and difference image have substantial emission in regions away from the true locations of flux. This is because the dirty beam sidelobes spread flux from the center of the image to other regions. CLEAN or RML would remove most of these features.
2. the difference image has fine-featured residuals in the center, corresponding to the edges of the antenna dishes and support structures. This is because the dirty beam has some Briggs weighting applied to it, and is closer to natural weighting than uniform weighting. This means that the spatial resolution of the dirty image is not as high as the original image, and thus high spatial frequency features, like the edges of the antennae, are not reproduced in the dirty image. Pushing the beam closer to uniform weighting would capture some of these finer structured features, but at the expense of higher thermal noise in the image.
3. the faint "halo" surrounding the antennas in the original image (the smooth blue sky and brown ground, in the actual JPEG) has been spatially filtered out of the dirty image. This is because this mock baseline distribution was generated for a more extended ALMA configuration without a sufficient number of short baselines.
