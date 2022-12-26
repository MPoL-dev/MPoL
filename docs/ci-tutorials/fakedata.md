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

# Making a Mock Dataset

In this tutorial, we'll explore how you might construct a mock dataset from a known sky brightness distribution. In many ways, this problem is already solved in a realistic manner by CASA's [simobserve](https://casadocs.readthedocs.io/en/latest/api/tt/casatasks.simulation.simobserve.html) task. However, by replicating the key parts of this process with MPoL framework, we can easily make direct comparisons to images produced using RML techniques.

In a nutshell, this process is works by
1. taking a known sky brightness distribution (i.e., a mock "true" image)
2. inserting it into an {class}`mpol.images.ImageCube`
3. using the {class}`mpol.fourier.NuFFT` to predict visibilities at provided $u,v$ locations
4. adding noise

The final two steps are relatively straightforward. The first two steps are conceptually simple but there are several technical concerns one should be aware of, which we'll cover now.

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
    caption: 'The ALMA antennas. Credit: ALMA (ESO/NAOJ/NRAO)

      '
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

```{code-cell} ipython3
from PIL import Image, ImageOps, ImageMath

im_raw = Image.open("alma.jpg")

# convert to greyscale
im_grey = ImageOps.grayscale(im_raw)
```

```{code-cell} ipython3
xsize, ysize = im_grey.size
```

```{code-cell} ipython3
im_grey
```

Now we'll need to make the image square. Depending on the image, we can either crop the longer dimension or pad the larger dimension. While we're doing this, we should also be thinking about the image edges.

Because the discrete Fourier transform is involved in taking an image to the visibility plane, we are making the assumption that the image is infinite and periodic in space beyond the field of view. i.e., it tiles to infinity. Therefore, to avoid introducing spurious spatial frequencies, it is a good idea to make sure that the edges of the image all have the same value. The simplest thing to do here is to taper the image edges such that they all go to 0. We can do this by multiplying by an apodization function, like the Hann window.

```{code-cell} ipython3
xhann = np.hanning(xsize)
yhann = np.hanning(ysize)
# each is already normalized to a max of 1
# so hann is also normed to a max of 1
# broadcast to 2D
hann = np.outer(yhann, xhann)

# scale to 0 - 255 and then convert to uint8
hann8 = np.uint8(hann * 255)
im_apod = Image.fromarray(hann8)
```

```{code-cell} ipython3
im_apod
```

Perform image math to multiply the taper against the greyscaled image

```{code-cell} ipython3
im_res = ImageMath.eval("a * b", a=im_grey, b=im_apod)
```

```{code-cell} ipython3
im_res
```

```{code-cell} ipython3
im_pad = ImageOps.pad(im_res, (1280,1280))
```

```{code-cell} ipython3
im_pad
```

```{code-cell} ipython3
im_small = im_pad.resize((500,500))
```

```{code-cell} ipython3
im_small
```

```{code-cell} ipython3
print(im_grey.mode)
```

In converting the JPEG image to greyscale (mode "L" or "luminance"), the Pillow library has reduced the color channel to a single axis with an 8-bit unsigned integer, which can take on the values from 0 to 255. More info on the modes is available [here](https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes).

```{code-cell} ipython3

```

Now that we have resized and tapered the image, we're ready to leave the Pillow library and work with numpy arrays and pytorch tensors. First we convert from a Pillow object to a numpy array

```{code-cell} ipython3
import numpy as np
a = np.array(im_small)
```

We can see that this array is now a 32-bit integer array (it was promoted during the ImageMath operation to save precision).

```{code-cell} ipython3
a
```

We will convert this array to a `float64` type and normalize its max value to 1.

```{code-cell} ipython3
b = a.astype("float64")
b = b/b.max()
```

Now, we can plot this array using matplotlib's `imshow`, as we might normally do with arrays of data

```{code-cell} ipython3
import matplotlib.pyplot as plt
plt.imshow(b, origin="upper")
```

But the main idea is that the values range from 0 to 255

```{code-cell} ipython3
a.dtype
```

```{code-cell} ipython3

```

This image is rectangular, with more pixels in the East-West direction compared to North-South. MPoL and the {class}`mpol.images.ImageCube` routines work (for now) under the assumption of square images. To rectify this situation, we will pad the North-South direction with zeros.

+++

This is most useful if you already have a real dataset, with real baseline distributions and noise weights. Alternatively, you could acquire some baseline distribution and noise distribution, possibly using CASA's simobserve.


How many pixels does it have?

The routine just takes an Image cube, u,v, weights and produces visibilities with noise.

But there's another concern about how to put the image cube in, right? I guess that's just a matter of matching the image cube to the size. You may want to pad the image, though. You probably also want to convolve it.

Here is where it would be helpful to have a note about how changing pixel size and image dimensions affects the uv coverage. There needs to be some match up between the image and the uv size.

Adjusting the `cell_size` changes the maximum spatial frequency that can be represented in the image. I.e., a smaller pixel cell size will enable an image to carry higher spatial frequencies.

Changing the number of pixels via `npix` will change the number of $u,v$ cells between 0 and the max spatial frequency.

Now, let's put this into a pytorch tensor, flip the directions, and insert it into an ImageCube.

```{code-cell} ipython3

```

We'll use the same u,v distribution and noise distribution from the mock dataset. The max baseline
