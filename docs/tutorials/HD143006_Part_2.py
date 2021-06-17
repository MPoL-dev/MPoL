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

# + [markdown] cell_id="00000-bfed870b-7d25-4899-b633-234d9e47dfa7" deepnote_cell_type="markdown"
# # HD143006 Tutorial Part 2
#
# This tutorial is a continuation of the [HD143006 Part 1](https://mpol-dev.github.io/MPoL/tutorials/HD143006_Part_1.html) tutorial and will follow the MPoL tutorials on [Optimization](https://mpol-dev.github.io/MPoL/tutorials/optimization.html) and [Cross Validation](https://mpol-dev.github.io/MPoL/tutorials/crossvalidation.html).It is assumed the users have familiarized themselves with these tutorials before hand.
#
# ### Loading Data
# Let's load the data as we did in the previous HD143006 tutorial ([Part 1](https://mpol-dev.github.io/MPoL/tutorials/HD143006_Part_1.html)) and create the MPoL Gridder object.
#
# *You can either download these two files (HD143006_continuum.fits and HD143006_continuum.npz) directly to your working directory, or use astropy to download them during run time.*

# + cell_id="00001-be94721b-eee2-4e2e-96e2-dc65b9fd4f5b" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=663 execution_start=1623447169390 source_hash="4f0f20f8" tags=[]
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.data import download_file

# downloading fits file
fname_F = download_file(
    "https://almascience.nrao.edu/almadata/lp/DSHARP/images/HD143006_continuum.fits",
    cache=True,
    pkgname="mpol",
)
# downloading extracted visibilities file
fname_EV = download_file(
    "https://zenodo.org/record/4904794/files/HD143006_continuum.npz",
    cache=True,
    pkgname="mpol",
)
# load extracted visibilities from npz file
dnpz = np.load(fname_EV)
uu = dnpz["uu"]
vv = dnpz["vv"]
weight = dnpz["weight"]
data = dnpz["data"]
# opening the fits file
dfits = fits.open(fname_F)
cdelt_scaling = dfits[0].header["CDELT1"] * 3600  # scaling [arcsec]
cell_size = abs(cdelt_scaling)  # [arcsec]
# close fits file
dfits.close()

# + cell_id="00008-d56e3afe-45cf-4fe5-a4e7-1803f28deec4" deepnote_cell_type="code" deepnote_to_be_reexecuted=false execution_millis=27 execution_start=1623441758279 source_hash="b76fed2d"
from mpol import gridding, coordinates

# creating Gridder object
coords = coordinates.GridCoords(cell_size=cell_size, npix=512)
gridder = gridding.Gridder(
    coords=coords,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data.real,  # seperating the real and imaginary values of our data
    data_im=data.imag,
)
# -

# We now have everything from the last tutorial loaded and can begin the process of Optimization and Cross Validation to improve our image quality.
#
# ### Getting the Dirty Image and Creating the Model
#
# First, we are going to get the dirty image from our gridder object. We will use the Briggs weighting scale and set `robust=0.0` here as these options lead to an already well optimized image (see [Part 1](https://mpol-dev.github.io/MPoL/tutorials/HD143006_Part_1.html)).

import torch

img, beam = gridder.get_dirty_image(weighting="briggs", robust=0.0, unit="Jy/arcsec^2")
# taking the dirty image and making it a tensor
dirty_image = torch.tensor(img.copy())

# Now we create the model.

from mpol.precomposed import SimpleNet

model = SimpleNet(coords=coords, nchan=gridder.nchan)

# ### Loss and Training Functions
#
# Now that we have our model and data, we need a loss function so we can help direct the Neural Network's learning. For this tutorial, will will use the MSELoss function which is part of the PyTorch library. We also want to create a Writer object so we can observe our Network's state at any point.
#
# *(i think this is correct about the writer, a little unsure)*

from torch.utils.tensorboard import SummaryWriter

loss = torch.nn.MSELoss()
writer = SummaryWriter()


# Now let us create a training function to train our SimpleNet


def train(model, dset, config, optimizer, loss_fn, writer):
    model.train()
    for iteration in range(config["epochs"]):
        optimizer.zero_grad()
        model.forward()
        sky_cube = model.icube.sky_cube
        loss = loss_fn(sky_cube, dirty_image)
        writer.add_scalar("loss", loss.item(), iteration)
        loss.backward()
        optimizer.step()
    # save the model
    torch.save(model.state_dict(), "model.pt")


# Now let's make our optimizer and our `config` variable. For the optimizer we will be following the [Cross Validation](https://mpol-dev.github.io/MPoL/tutorials/crossvalidation.html) tutorial and for the `config` we will start with a non-agressive learning rate and a low number of epochs.

config = {"lr": 0.5, "epochs": 500}
optim = torch.optim.Adam(model.parameters(), lr=config["lr"])

# Finally, lets run our training function and then plot the results.

train(model, dirty_image, config, optim, loss, writer)

# +
fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

im = ax[0].imshow(
    np.squeeze(dirty_image.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
)

im = ax[1].imshow(
    np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
)

ax[0].set_xlim(left=0.75, right=-0.75)
ax[0].set_ylim(bottom=-0.75, top=0.75)
ax[0].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[0].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
ax[0].set_title("MPoL Dirty Image")
ax[1].set_xlim(left=0.75, right=-0.75)
ax[1].set_ylim(bottom=-0.75, top=0.75)
ax[1].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[1].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
ax[1].set_title("MPoL Optimized Dirty Image")
plt.tight_layout()

# +
# EDIT/MOVE this is the loss function

# %load_ext tensorboard
# %tensorboard --logdir logs

# NOTE- The tensorboard info can also be accessed from the terminal using
# tensorboard --logdir=runs
# -

# Will be changing code around and updating this a little, just wanted to see the as-is results from optimization loop
#

# ### Cross Validation
#
# Now we will move into the realm of Cross Validation. To do this we will be utilizing the [Ray[Tune]](https://docs.ray.io/en/master/tune/index.html) python package for hyperparameter tuning. In order to get the best fit, we will be modifying our `train` function to encorperate a stronger loss function. We will import this from `mpol.losses`. We also need the `mpol.connectors` package because ....?.... Let us do that now.

from mpol import losses, connectors


def train(model, dataset, optimizer, config, writer=None, report=False, logevery=50):
    model.train()
    for iteration in range(config["epochs"]):
        optimizer.zero_grad()
        vis = model.forward()
        sky_cube = model.icube.sky_cube
        # computing loss through MPoL loss function with more parameters
        loss = (
            losses.nll_gridded(vis, dataset)
            + config["lambda_sparsity"] * losses.sparsity(sky_cube)
            + config["lambda_TV"] * losses.TV_image(sky_cube)
        )
        residuals = connectors.GriddedResidualConnector(vis.fcube, dataset)
        residuals.forward()

        if (iteration % logevery == 0) and writer is not None:
            writer.add_scalar("loss", loss.item(), iteration)
            writer.add_figure("image", log_figure(model, residuals), iteration)

        loss.backward()
        optimizer.step()
    if report:
        tune.report(loss=loss.item())

    return loss.item()


# from mpol import losses, connectors
# def train(model, dataset, optimizer, config, writer=None, report=False, logevery=50):
#     model.train()
#     residuals = connectors.GriddedResidualConnector(model.fcube, dataset)
#     for iteration in range(config["epochs"]):
#         optimizer.zero_grad()
#         vis = model.forward()
#         sky_cube = model.icube.sky_cube
#         # computing loss through MPoL loss function with more parameters
#         loss = (
#             losses.nll_gridded(vis, dataset)
#             + config["lambda_sparsity"] * losses.sparsity(sky_cube)
#             + config["lambda_TV"] * losses.TV_image(sky_cube)
#         )
#
#         if (iteration % logevery == 0) and writer is not None:
#             writer.add_scalar("loss", loss.item(), iteration)
#             writer.add_figure("image", log_figure(model, residuals), iteration)
#
#         loss.backward()
#         optimizer.step()
#     if report:
#         tune.report(loss=loss.item())
#
#     return loss.item()

# Just like in the [Cross Validation tutorial](https://mpol-dev.github.io/MPoL/tutorials/crossvalidation.html) we will need a `test` function and a `cross_validate` function. We impliment these below.


def test(model, dataset):
    model.eval()
    vis = model.forward()
    loss = losses.nll_gridded(vis, dataset)
    return loss.item()


def cross_validate(model, config, k_fold_datasets, MODEL_PATH, writer=None):
    test_scores = []

    for k_fold, (train_dset, test_dset) in enumerate(k_fold_datasets):
        # reset model
        model.load_state_dict(model_state)

        # create a new optimizer for this k_fold
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        # train for a while
        train(model, train_dset, optimizer, config, writer=writer, report=True)
        # evaluate the test metric
        test_scores.append(test(model, test_dset))

    # aggregate all test scores and sum to evaluate cross val metric
    test_score = np.sum(np.array(test_scores))

    # log to ray tune
    tune.report(cv_score=test_score)

    return test_score


# Now we will impliment hyperparameter tuning with Ray\[Tune\]. To do this, we need a function that only takes a `config` variable as input. We will call this `trainable`. Using Ray\[Tune\] is rather straight forward. You set up a `tune.run()` object with your function and the parameters and it will calculate the best parameters (this needs to be worded better).


def trainable(config):
    cross_validate(model, config, k_fold_datasets, model_state)


from mpol import datasets

dartboard = datasets.Dartboard(coords=coords)
dataset = gridder.to_pytorch_dataset()
# create cross validator using this "dartboard"
k = 5
cv = datasets.KFoldCrossValidatorGridded(dataset, k, dartboard=dartboard, npseed=42)
k_fold_datasets = [(train, test) for (train, test) in cv]

from ray import tune
import ray

# making sure that we don't initialize ray if its already initialized
ray.shutdown()
ray.init()
MODEL_PATH = "./model.pt"
model_state = torch.load(MODEL_PATH)
analysis = tune.run(
    trainable,
    config={
        "lr": 0.3,
        "lambda_sparsity": tune.loguniform(1e-8, 1e-4),
        "lambda_TV": tune.loguniform(1e-4, 1e1),
        "epochs": 1000,
    },
    resources_per_trial={"cpu": 3},
)
print("Best config: ", analysis.get_best_config(metric="cv_score", mode="min"))

# +
fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

im = ax[0].imshow(
    np.squeeze(dirty_image.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
)

im = ax[1].imshow(
    np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
)

ax[0].set_xlim(left=0.75, right=-0.75)
ax[0].set_ylim(bottom=-0.75, top=0.75)
ax[0].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[0].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
ax[0].set_title("MPoL Dirty Image")
ax[1].set_xlim(left=0.75, right=-0.75)
ax[1].set_ylim(bottom=-0.75, top=0.75)
ax[1].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[1].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
ax[1].set_title("MPoL Optimized Dirty Image")
plt.tight_layout()
# -

torch.save(model.state_dict(), "model1.pt")
