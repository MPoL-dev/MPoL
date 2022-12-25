---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
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

# Intro to MPoL Optimization

In this tutorial, we'll construct an optimization loop demonstrating how we can use MPoL to synthesize a basic image. We'll continue with the dataset described in the [Gridding and Diagnostic Images](gridder.md) tutorial.

## Gridding recap
Let's set up the {class}`~mpol.gridding.Gridder` and {class}`~mpol.coordinates.GridCoords` objects as before

```{code-cell}
import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.utils.data import download_file
from IPython.display import SVG, display
```

```{code-cell}
from mpol import coordinates, gridding, losses, precomposed, utils
```

```{code-cell}
# load the mock dataset of the ALMA logo
fname = download_file(
    "https://zenodo.org/record/4930016/files/logo_cube.noise.npz",
    cache=True,
    show_progress=True,
    pkgname="mpol",
)

# this is a multi-channel dataset... for demonstration purposes we'll use
# only the central, single channel
chan = 4
d = np.load(fname)
uu = d["uu"][chan]
vv = d["vv"][chan]
weight = d["weight"][chan]
data = d["data"][chan]
data_re = np.real(data)
data_im = np.imag(data)
```

```{code-cell}
# define the image dimensions, as in the previous tutorial
coords = coordinates.GridCoords(cell_size=0.005, npix=800)
gridder = gridding.Gridder(
    coords=coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data_re,
    data_im=data_im,
)
```

## The PyTorch dataset

```{margin} Cell-averaging
The visibility averaging step performed by the gridder is a weighted average that is numerically equivalent to "uniform" weighting of the visibilities; this does not mean that MPoL or RML only produces images that have "uniform" weighting, however. The gridder also propagates the uncertainties from the individual visibilities to an uncertainty on the averaged visibility cell. When MPoL *forward-models* the visibility dataset and evaluates model image against the data, these uncertainties are used in a likelihood function, which is combined with priors/regularizers and the numerical results will be the same whether or not the likelihood function is computed using the gridded or [ungridded](loose-visibilities.md) visibilities. By contrast, dirty images are a direct inverse Fourier transform of the gridded visibility data and depend on whether the visibilities were weighted with uniform, natural, or Briggs weighting schemes.
```

Now we will export the visibilities to a PyTorch dataset to use in the imaging loop. The {meth}`mpol.gridding.Gridder.to_pytorch_dataset` routine performs a weighted average all of the visibilities to the Fourier grid cells and exports the visibilities to cube-like PyTorch tensors. To keep things simple in this tutorial, we are only using a single channel. But you could just as easily export a multi-channel dataset. Note that the {meth}`~mpol.gridding.Gridder.to_pytorch_dataset` routine automatically checks the visibility scatter and raises a ``RuntimeError`` if the empirically-estimated scatter exceeds that expected from the provided dataset weights. For more information, see the end of the [Gridding and Diagnostic Images Tutorial](gridder.md).

In the following [tutorial on the NuFFT](loose-visibilities.md), we'll explore an alternate MPoL layer that avoids gridding the visibilities all together. This approach may be more accurate for certain applications, but is usually slower to execute than the gridding approach described in this tutorial. For that reason, we recommend starting with the default gridding approach and only moving to the NuFFT layers once you are reasonably happy with the images you are getting.

```{code-cell}
dset = gridder.to_pytorch_dataset()
print("this dataset has {:} channel".format(dset.nchan))
```

## Building an image model

MPoL provides "modules" to build and optimize complex imaging workflows, not dissimilar to how a deep neural network might be constructed. We've bundled the most common modules for imaging together in a {class}`mpol.precomposed.SimpleNet` meta-module, which we'll use here.

This diagram shows how the primitive modules, like {class}`mpol.images.BaseCube`, {class}`mpol.images.ImageCube`, etc... are connected together to form {class}`mpol.precomposed.SimpleNet`. In this workflow, the pixel values of the {class}`mpol.images.BaseCube` are the core model parameters representing the image. More information about all of these components is available in the {ref}`API documentation <api-reference-label>`.

```{mermaid} ../_static/mmd/src/SimpleNet.mmd
```

It isn't necessary to construct a meta-module to do RML imaging with MPoL, though it often helps organize your code. If we so desired, we could connect the individual modules together ourselves ourselves following the SimpleNet source code as an example ({class}`mpol.precomposed.SimpleNet`) and swap in/out modules as we saw fit.

We then initialize SimpleNet with the relevant information

```{code-cell}
rml = precomposed.SimpleNet(coords=coords, nchan=dset.nchan)
```

## Breaking down the training loop

Our goal for the rest of the tutorial is to set up a loop that will

1. evaluate the current model against a loss function
2. calculate the gradients of the loss w.r.t. the model
3. advance the model parameters in the direction to minimize the loss function

We'll start by creating the optimizer

```{code-cell}
optimizer = torch.optim.SGD(rml.parameters(), lr=8000.0)
```

The role of the optimizer is to advance the parameters (in this case, the pixel values of the {class}`mpol.images.BaseCube` using the gradient of the loss function with respect to those parameters. PyTorch has many different [optimizers](https://pytorch.org/docs/stable/optim.html#module-torch.optim) available, and it is worthwhile to try out some of the different ones. Stochastic Gradient Descent (SGD) is one of the simplest, so we’ll start here. The `lr` parameter is the 'learning rate,' or how ambitious the optimizer should be in taking descent steps. Tuning this requires a bit of trial and error: you want the learning rate to be small enough so that the algorithm doesn’t diverge but large enough so that the optimization completes in a reasonable amount of time.


## Loss functions
In the parlance of the machine learning community, one defines "loss" functions comparing models to data. For regularized maximum likelihood imaging, the most fundamental loss function we'll use is the {func}`mpol.losses.loss_fn` or the $\chi^2$ value comparing the model visibilities to the data visibilities. For this introductory tutorial, we'll use only the data likelihood loss function to start, but you should know that because imaging is an ill-defined inverse problem, this is **not a sufficient constraint** by itself. In later tutorials, we will apply regularization to narrow the set of possible images towards ones that we believe are more realistic. The {mod}`mpol.losses` module contains several loss functions currently popular in the literature, so you can experiment to see which best suits your application.

## Gradient descent

Let's walk through how we calculate a loss value and optimize the parameters. To start, let's [examine the parameters of the model](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

```{code-cell}
rml.state_dict()
```

These are the default values that were used to initialize the {class}`mpol.images.BaseCube` component of the {class}`mpol.precomposed.SimpleNet`.

For demonstration purposes, lets access and plot the base cube with matplotlib. In a normal workflow you probably won't need to do this, but to access the basecube in sky orientation, we do

```{code-cell}
bcube_pytorch = utils.packed_cube_to_sky_cube(rml.bcube.base_cube)
```

``bcube`` is still a PyTorch tensor, but matplotlib requires numpy arrays. To convert back, we need to first ["detach" the computational graph](https://stackoverflow.com/questions/63582590/why-do-we-call-detach-before-calling-numpy-on-a-pytorch-tensor) from the PyTorch tensor (used to propagate gradients) and then call the numpy conversion routine.

```{code-cell}
bcube_numpy = bcube_pytorch.detach().numpy()
print(bcube_numpy.shape)
```

lastly, we remove the channel dimension to plot the 2D image using ``np.squeeze``

```{code-cell}
fig, ax = plt.subplots(nrows=1)
im = ax.imshow(
    np.squeeze(bcube_numpy),
    origin="lower",
    interpolation="none",
    extent=rml.icube.coords.img_ext,
)
plt.xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
plt.ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
plt.colorbar(im)
```

A blank image is not that exciting, but hopefully this demonstrates the state of the parameters at the start of optimization.

Because we'll want to compute a clean set of gradient values in a later step, we "zero out" any gradients attached to the tensor components so that they aren't counted twice.

```{code-cell}
rml.zero_grad()
```

Most modules in MPoL are designed to work in a "feed forward" manner, which means base parameters are processed through the network to predict model visibilites for comparison with data. We can calculate the full visibility cube corresponding to the current pixel values of the {class}`mpol.images.BaseCube`.

```{code-cell}
vis = rml.forward()
print(vis)
```

Of course, these aren't that exciting since they just reflect the constant value image.

But, exciting things are about to happen! We can calculate the loss between these model visibilities and the data

```{code-cell}
# calculate a loss
loss = losses.nll_gridded(vis, dset)
print(loss.item())
```

and then we can calculate the gradient of the loss function with respect to the parameters

```{code-cell}
loss.backward()
```

We can even visualize what the gradient of the {class}`mpol.images.BaseCube` looks like (using a similar ``.detach()`` call as before)

```{code-cell}
fig, ax = plt.subplots(nrows=1)
im = ax.imshow(
    np.squeeze(
        utils.packed_cube_to_sky_cube(rml.bcube.base_cube.grad).detach().numpy()
    ),
    origin="lower",
    interpolation="none",
    extent=rml.icube.coords.img_ext,
)
plt.xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
plt.ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
plt.colorbar(im)
```

The gradient image points in the direction of lower loss values. So the final step is to add the gradient image to the base image in order to advance base parameters in the direction of the minimum loss value. This process is called gradient descent, and can be extremely useful for optimizing large dimensional parameter spaces (like images). The optimizer carries out the addition of the gradient

```{code-cell}
optimizer.step()
```

We can see that the parameter values have changed

```{code-cell}
rml.state_dict()
```

as has the base image

```{code-cell}
fig, ax = plt.subplots(nrows=1)
im = ax.imshow(
    np.squeeze(utils.packed_cube_to_sky_cube(rml.bcube.base_cube).detach().numpy()),
    origin="lower",
    interpolation="none",
    extent=rml.icube.coords.img_ext,
)
plt.xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
plt.ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
plt.colorbar(im)
```

## Iterating the training Loop

Now that we've covered how to use gradient descent to optimize a set of image parameters, let's wrap these steps into a training loop and iterate a few hundred times to converge to a final product.

In addition to the steps just outlined, we'll also track the loss values as we optimize.

```{code-cell}
%%time

loss_tracker = []

for i in range(300):
    rml.zero_grad()

    # get the predicted model
    vis = rml.forward()

    # calculate a loss
    loss = losses.nll_gridded(vis, dset)

    loss_tracker.append(loss.item())

    # calculate gradients of parameters
    loss.backward()

    # update the model parameters
    optimizer.step()
```

```{code-cell}
fig, ax = plt.subplots(nrows=1)
ax.plot(loss_tracker)
ax.set_xlabel("iteration")
ax.set_ylabel("loss")
```

and we see that we've reasonably converged to a set of parameters without much further improvement in the loss value.

All of the method presented here can be sped up using GPU acceleration on certain Nvidia GPUs. To learn more about this, please see the {ref}`GPU Setup Tutorial <gpu-reference-label>`.

## Visualizing output

Let's visualize the final product. The bounds for `matplotlib.pyplot.imshow` are available in the `img_ext` parameter.

```{code-cell}
# let's see what one channel of the image looks like
fig, ax = plt.subplots(nrows=1)
im = ax.imshow(
    np.squeeze(rml.icube.sky_cube.detach().numpy()),
    origin="lower",
    interpolation="none",
    extent=rml.icube.coords.img_ext,
)
plt.xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
plt.ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
plt.colorbar(im)
```

## Wrapup

And there you have it, an image optimized to fit the data. To be honest, the results aren't great---that's because we've used minimal regularization in the form of the functional basis set we chose that automatically enforced image positivity (see the {class}`mpol.images.BaseCube` documentation). Otherwise, our only contribution to the loss function is the data likelihood. This means it's easy for the lower signal-to-noise visibilities at longer baselines to dominate the image appearance (not unlike "uniform"ly weighted images).

In the following tutorials we'll examine how to set up additional regularizer terms that can yield more desireable image characteristics.

Hopefully this tutorial has demonstrated the core concepts of synthesizing an image with MPoL. If you have any questions about the process, please feel free to reach out and start a [GitHub discussion](https://github.com/MPoL-dev/MPoL/discussions). If you spot a bug or have an idea to improve these tutorials, please raise a [GitHub issue](https://github.com/MPoL-dev/MPoL/issues) or better yet submit a pull request.
