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

Great, we now have a square, apodized image. The only thing is that a 1280 x 1280 image is still a bit too many pixels for most ALMA observations. I.e., the spatial resolution or "beam size" of most ALMA observations is such that for any single-pointing observation, we wouldn't need this many pixels to represent the full information content of the image. Therefore, let's resize the image to be a bit smaller.

```{code-cell} ipython3
npix = 500
im_small = im_pad.resize((npix,npix))
```

```{code-cell} ipython3
im_small
```

## Exporting to a PyTorch tensor

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

Now, we'll convert the numpy array to a PyTorch tensor

```{code-cell} ipython3
import torch
img_tensor = torch.tensor(d.copy())
```

And finally, we'll shift the tensor from a "Sky Cube" to a "Packed Cube" as the {class}`~mpol.images.ImageCube` expects

```{code-cell} ipython3
from mpol import utils
img_tensor_packed = utils.sky_cube_to_packed_cube(img_tensor)
```

## Initializing {class}`~mpol.images.ImageCube`

Now let's settle on how big

Here is where it would be helpful to have a note about how changing pixel size and image dimensions affects the uv coverage. There needs to be some match up between the image and the uv size.

Adjusting the `cell_size` changes the maximum spatial frequency that can be represented in the image. I.e., a smaller pixel cell size will enable an image to carry higher spatial frequencies.

Changing the number of pixels via `npix` will change the number of $u,v$ cells between 0 and the max spatial frequency.

We already defined `npix` when we performed the resize operation.

```{code-cell} ipython3
cell_size = 0.03 # arcsec

from mpol.images import ImageCube
image = ImageCube(cell_size=cell_size, npix=npix, nchan=1, cube=img_tensor_packed)
```

```{code-cell} ipython3
# double check it went in correctly
# plt.imshow(np.squeeze(utils.packed_cube_to_sky_cube(image.forward()).detach().numpy()), origin="lower")
```

## Getting baseline distributions

This is most useful if you already have a real dataset, with real baseline distributions and noise weights. Alternatively, you could acquire some baseline distribution and noise distribution, possibly using CASA's simobserve.

In this example, we'll just use the baseline distribution from the mock dataset we've used in many of the tutorials. You can see a plot of it in the [Gridding and Diagnostic Images](gridder.md) tutorial. We'll only need the $u,v$ and weight arrays.

```{code-cell} ipython3
from astropy.utils.data import download_file

# load the mock dataset of the ALMA logo
fname = download_file(
    "https://zenodo.org/record/4930016/files/logo_cube.noise.npz",
    cache=True,
    show_progress=True,
    pkgname="mpol",
)

d = np.load(fname)
uu = d["uu"]
vv = d["vv"]
weight = d["weight"]
```

```{code-cell} ipython3
max_uv = np.max(np.array([uu,vv]))
max_cell_size = utils.get_maximum_cell_size(max_uv)
print("The maximum cell_size that will still Nyquist sample the spatial frequency represented by the maximum u,v value is {:.2f} arcseconds".format(max_cell_size))
```

```{code-cell} ipython3
# will have the same shape as the uu, vv, and weight inputs
data_noise, data_noiseless = make_fake_data(image, u, v, weight)
```

How many pixels does it have?

The routine just takes an Image cube, u,v, weights and produces visibilities with noise.



Now, let's put this into a pytorch tensor, flip the directions, and insert it into an ImageCube.

```{code-cell} ipython3

```

We'll use the same u,v distribution and noise distribution from the mock dataset. The max baseline


## Making the mock dataset


Now you could save this to disk, for example



## Verifying the mock dataset

To make sure the whole process worked OK, we'll load the visibilities and then make a dirty image.
