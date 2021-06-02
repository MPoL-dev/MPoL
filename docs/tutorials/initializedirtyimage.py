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

# ## Initializing Model with the Dirty Image
#
# In this tutorial we will be going over how to optimize the model based off of the dirty image. It has a low loss value (maximal likelihood) and so this allows the optimization to be much quicker. The problem with the dirty image is that certain model paramterizations enforce image positivity while the dirty image usually contains regions of negative flux. So in this tutorial we will be optimizing the model to be similar to the dirty image (phrasing could be better here). We will also be going over saving and loading the model. After saving and loading it, it can be then optimized against the visibility data to complete it (though this will not be done in this tutorial). We will be using the dataset of the ALMA logo first used in the Gridding and Diagnostic Images tutorial.
#
#

# ### Image Setup
#
# Here we will be setting up the ALMA Logo image dataset and displaying it. Consult the Gridding and Diagnostic Images Tutorial for reference.

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpol import gridding, coordinates, precomposed, losses, utils
from astropy.utils.data import download_file
from torch.utils.tensorboard import SummaryWriter

# In this tutorial we will be saving and loading the model we use and it is important to note that when saving and loading a model the cell_size, nchan, and npix are the same between the model and the coordinates if used in another document.

# +
# load the mock dataset of the ALMA logo
fname = download_file(
    "https://zenodo.org/record/4498439/files/logo_cube.npz",
    cache=True,
    show_progress=True,
    pkgname="mpol",
)

# this is a multi-channel dataset... but for demonstration purposes we'll use
# only the central, single channel
chan = 4
d = np.load(fname)
uu = d["uu"][chan]
vv = d["vv"][chan]
weight = d["weight"][chan]
data_re = d["data_re"][chan]
data_im = -d["data_im"][
    chan
]  # we're converting from CASA convention to regular TMS convention by complex conjugating the visibilities

# define the image dimensions, making sure they are big enough to fit all
# of the expected emission
coords = coordinates.GridCoords(
    cell_size=0.03, npix=180
)  # The smaller the cell size and the larger the npix can greatly increase run time
gridder = gridding.Gridder(
    coords=coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data_re,
    data_im=data_im,
)

# export to PyTorch dataset
dset = gridder.to_pytorch_dataset()
# -

# Show the dirty image
img, beam = gridder.get_dirty_image(weighting="briggs", robust=0.0, unit="Jy/beam")
kw = {"origin": "lower", "extent": gridder.coords.img_ext}
fig, ax = plt.subplots(ncols=1)
snp = ax.imshow(np.squeeze(img), **kw)
ax.set_title("image")
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
plt.colorbar(snp)

# ### Model and Optimization Setup
#
# Here we will be setting up the optimizer and the image model (RML). If this is unfamiliar please reference the Optimization tutorial. The initial parameters of the model are also displayed and can be contrasted with them after the optimization loop. }

dirty_image = torch.tensor(img.copy())  # turns it into a pytorch tensor
rml = precomposed.SimpleNet(coords=coords, nchan=dset.nchan)
optimizer = torch.optim.SGD(
    rml.parameters(), lr=500.0
)  # multiple different possiple optimizers
rml.state_dict()  # parameters of the model


# ### Loss Function

# The loss function that will be used to optimize the initial part of the image is the pixel-to-pixel L2 norm (also known as the Euclidian Norm). It calculates the loss based off of the image-plane distance between the dirty image and the state of the ImageCube in order to make the state of the ImageCube closer to the dirty image.
#
# The L2 Norm (Euclidian Norm) is based off of the euclidian distance function and it takes the form of: $ L_{oss} = \sqrt{\sum_{i=1}^{N} (y_i - x_i)^2}  $. Pytorch provides a loss function for mean squared error which is the squared L2 norm so we will be using the squareroot of that.


# ### Training Loop
#
# Now we will be starting a trainining loop using this loss function to optimize our parameters. There will also be a loss tracker to see how loss falls off over time.

# +
# %%time


loss_tracker = []
for iteration in range(50):

    optimizer.zero_grad()

    rml.forward()

    sky_cube = rml.icube.sky_cube

    lossfunc = torch.nn.MSELoss(
        reduction="sum"
    )  # the MSELoss calculates mean squared error (squared L2 norm), so we take the sqrt of it
    loss = (
        lossfunc(sky_cube, dirty_image)
    ) ** 0.5  # https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html

    loss_tracker.append(loss.item())
    loss.backward()
    optimizer.step()

torch.save(rml.state_dict(), "dirty_image_model.pt")
# -

# We also can save the state of the model. The [torch.save(model.state_dict(), PATH)](https://pytorch.org/tutorials/beginner/saving_loading_models.html) will save the state.dict() which are the model's learned parameters.
#

# ### Results

fig, ax = plt.subplots(nrows=1)
ax.plot(loss_tracker)
ax.set_xlabel("iteration")
ax.set_ylabel("loss")

# The results are seen here with the state_dict() having changed and the image now more closely resembling the dirty image, though the dirty image contains negative values which this does not so they can never be completely the same.

fig, ax = plt.subplots(nrows=1)
im = ax.imshow(
    np.squeeze(rml.icube.sky_cube.detach().numpy()),
    origin="lower",
    interpolation="none",
    extent=rml.icube.coords.img_ext,
)
plt.colorbar(im)

rml.state_dict()

# ### Loading the Model
#
# To demonstrate the saving and loading of the model here we will be resetting the parameters of the model and then reloading them.

rml = precomposed.SimpleNet(coords=coords)
rml.state_dict()  # parameters of the model

# Here you can clearly see the state_dict() returning to its original state from the beginning of the tutorial and then once we reload it, it reverts to the values it was when we saved it.

rml.load_state_dict(torch.load("dirty_image_model.pt"))
rml.state_dict()  # parameters of the model

# To continue and optimize our model against the visibility data the gridded visibilitied need to have uniform weighting

gridder._grid_visibilities(weighting="uniform")
img, beam = gridder.get_dirty_image(weighting="uniform")
dataset = gridder.to_pytorch_dataset()  # creating a new dataset

# ### Conclusion
#
# This tutorial shows how to work towards a better starting point for RML optimization by using an image with maximal likelihood but RML optimization does not need to start from here and can start from a blank image. This is unlike a deconvolution algorithm such as the CASA tclean. The training loop done here is just the set up and the model/image still needs further regularization and optimization against the visibility data.
#
# https://casa.nrao.edu/docs/taskref/tclean-task.html https://casa.nrao.edu/casadocs/casa-6.1.0/imaging/synthesis-imaging/deconvolution-algorithms#:~:text=Deconvolution%20refers%20to%20the%20process,spread%2Dfunction%20of%20the%20instrument.
