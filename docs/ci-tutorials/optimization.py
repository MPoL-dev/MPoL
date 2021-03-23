# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.0
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


# # Optimization Loop
#
# We'll continue from where we left off in the [Gridding and Diagnostic Images](../gridder) tutorial and construct a basic optimization loop for RML imaging.

import torch
import numpy as np
import matplotlib.pyplot as plt
import mpol
from mpol import gridding, coordinates, precomposed, losses
import requests
from astropy.utils.data import download_file
import os

# +
fname = download_file(
    "https://zenodo.org/record/4498439/files/logo_cube.npz",
    cache=True,
    show_progress=True,
    pkgname="mpol",
)

chan = 4
d = np.load(fname)
uu = d["uu"][chan]
vv = d["vv"][chan]
weight = d["weight"][chan]
data_re = d["data_re"][chan]
data_im = -d["data_im"][
    chan
]  # we're converting from CASA convention to regular TMS convention by complex conjugating the visibilities
# -

coords = gridding.GridCoords(cell_size=0.005, npix=800)
gridder = gridding.Gridder(
    coords=coords, uu=uu, vv=vv, weight=weight, data_re=data_re, data_im=data_im,
)
gridder.grid_visibilities(weighting="uniform")

# Now let's export to a PyTorch dataset. Note that we needed to do the gridding with "uniform".

dset = gridder.to_pytorch_dataset()

dset.nchan

# ## Image Model


# Let's build a simple RML model

# ![SimpleNet](../_static/mmd/build/SimpleNet.svg)

# +
from IPython.display import SVG, Image, display

display(SVG(filename="../_static/mmd/SimpleNet.svg"))
# -

# ## Optimizer

# +
# %%time

# set everything up to run on a single channel
nchan = dset.nchan
rml = precomposed.SimpleNet(coords=coords, nchan=nchan, griddedDataset=dset)

optimizer = torch.optim.SGD(rml.parameters(), lr=100.0, momentum=0.1)

loss_log = []

for i in range(200):
    rml.zero_grad()

    # get the predicted model
    model_visibilities = rml.forward()

    # calculate a loss
    loss = losses.nll(
        model_visibilities, dset.vis_indexed, dset.weight_indexed
    )  # + 1e-4 * losses.loss_fn_TV_image(rml.icube.sky_cube)

    loss_log.append(loss.item())

    # calculate gradients of parameters
    loss.backward()

    # store a residual vis and image
    # from RML.vis and dset.vis

    # update the model parameters
    optimizer.step()
# -

fig, ax = plt.subplots(nrows=1)
ax.plot(loss_log)
ax.set_xlabel("iteration")
ax.set_ylabel("loss")

import copy

trained_nll = copy.deepcopy(rml.state_dict())
img_nll = np.squeeze(rml.icube.sky_cube.detach().numpy())

trained_nll

# The nll image is not necessarily more conservative! All images involve choices, and in some ways one without a prior is actually a more radical choice, since we ignore image constraints we believe should function. Not quite a dirty image either, since 1) dirty image is just one of many maximal likelihood images 2) we have imposed image postivitiy

# let's see what one channel of the image looks like
fig, ax = plt.subplots(nrows=1)
im = ax.imshow(
    img_nll, origin="lower", interpolation="none", extent=rml.icube.coords.img_ext,
)
plt.colorbar(im)

# +
# set everything up to run on a single channel
nchan = dset.nchan
rml.load_state_dict(trained_nll)
loss_log = []
optimizer = torch.optim.SGD(rml.parameters(), lr=3000, momentum=0.3)

for i in range(500):
    rml.zero_grad()

    # get the predicted model
    model_visibilities = rml.forward()

    # calculate a loss
    loss = losses.nll(
        model_visibilities, dset.vis_indexed, dset.weight_indexed
    ) + 1e-9 * losses.loss_fn_sparsity(rml.icube.sky_cube) + 1e-4 * losses.loss_fn_TV_image(rml.icube.sky_cube)

    loss_log.append(loss.item())

    # calculate gradients of parameters
    loss.backward()

    # store a residual vis and image
    # from RML.vis and dset.vis

    # update the model parameters
    optimizer.step()
# -

iter1 = copy.deepcopy(rml.state_dict())

# As we’ll see in a moment, this optimizer will advance the parameters (in this case, the pixel values of the image cube) based upon the gradient of the loss function with respect to those parameters. PyTorch has many different [optimizers](https://pytorch.org/docs/stable/optim.html#module-torch.optim) available, and it would be worthwhile to try out some of the different ones. Stochastic Gradient Descent (SGD) is one of the simplest, so we’ll start here. The `lr` parameter is the 'loss rate,' or how ambitious the optimizer should be in taking descent steps. Tuning this requires a bit of trial and error: you want the loss rate to be small enough so that the algorithm doesn’t diverge but large enough so that the optimization completes in a reasonable amount of time.
#
# ## Losses
# In the parlance of the machine learning community, one can define loss functions against the model image and visibilities. For regularized maximum likelihood imaging, one key loss function that we are interested in is the data likelihood (`mpol.losses.loss_fn()`), which is just the $\chi^2$ of the visibilities. Because imaging is an ill-defined inverse problem, however, the visibility likelihood function is not sufficient. We also need to apply regularization to narrow the set of possible images towards ones that we believe are more realistic. The mpol.losses module contains several loss functions currently popular in the literature, so you can experiment to see which best suits your application.
#
# ## Training loop
# Next, we’ll set up a loop that will
#
# 1. evaluate the current model (i.e., the image cube) against the loss functions
# 2. calculate the gradients of the loss w.r.t. the model
# 3. advance the model so as to minimize the loss
#
# Here is a minimal loop that will accomplish this and track the value of the loss with each iteration.

# It is an excellent idea to track and plot diagnostics like the loss values while optimizing. This will help gain intuition for how the penalty terms (the scale factor in front of the sparsity regularization) affect the image quality. You can also query and save the image cube values and RFFT output during optimization as well.
#
# Moreover, you can compose many intricate optimization strategies using the tools available in PyTorch.

fig, ax = plt.subplots(nrows=1)
ax.plot(loss_log)
ax.set_xlabel("iteration")
ax.set_ylabel("loss")

# ## Saving output
# When you are finished optimizing, you can save the output.
#
# Image bounds for `matplotlib.pyplot.imshow` are available in `model.extent`.

# + id="YiFVw5B8cjL-"
# let's see what one channel of the image looks like
fig, ax = plt.subplots(nrows=1)
im = ax.imshow(
    np.squeeze(rml.icube.sky_cube.detach().numpy()),
    origin="lower",
    interpolation="none",
    extent=rml.icube.coords.img_ext,
)
plt.colorbar(im)
# -

