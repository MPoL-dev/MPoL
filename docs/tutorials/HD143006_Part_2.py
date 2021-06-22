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
# This tutorial is a continuation of the [HD143006 Part 1](https://mpol-dev.github.io/MPoL/tutorials/HD143006_Part_1.html) tutorial and will follow the MPoL tutorials on [Optimization](optimization.html), [Initalizing with the Dirty Image](initializedirtyimage.html), and [Cross Validation](crossvalidation.html). It is assumed the users have familiarized themselves with these tutorials before hand.
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
from torch.utils import tensorboard

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
# First, we are going to get the dirty image from our gridder object. We will use the Briggs weighting scale and set `robust=0.0` here as these options lead to an already well optimized image (see [Part 1](HD143006_Part_1.html)).

import torch

img, beam = gridder.get_dirty_image(weighting="briggs", robust=0.0, unit="Jy/arcsec^2")
# taking the dirty image and making it a tensor
dirty_image = torch.tensor(img.copy())

# Now we create the RML model.

from mpol.precomposed import SimpleNet

model = SimpleNet(coords=coords, nchan=gridder.nchan)

# ### Initializing Model with the Dirty Image
#
# We now have our model and data, but before we set out trying to optimize the image we should create a better starting point for our future optimization loops. A good idea for the starting point is the dirty image, since it is already a maximum likelihood fit to the data. The problem with this is that the dirty image containes negative flux pixels, while we impose the requirement that our sources must have all positive flux values. Our solution then is to optimize the RML model to become as close to the dirty image as possible (while retaining image positivity).
#
# We also want to create a Writer object so we can observe our Network's state at any point.
#
# *(i think this is correct about the writer, a little unsure)*

from torch.utils.tensorboard import SummaryWriter  # setting up the writer
import os

logs_base_dir = "./logs"
writer = SummaryWriter(logs_base_dir)
os.makedirs(logs_base_dir, exist_ok=True)
# %load_ext tensorboard
# uncomment above line in jupyter notebook
# still working on what to do with that for .py


# Now we will create our training loop using a [loss function](../api.html#module-mpol.losses) (here we use the mean squared error between the RML model image pixel fluxes and the dirty image pixel flues) and an [optimizer](https://pytorch.org/docs/stable/optim.html#module-torch.optim). MPoL and Pytorch contain many different optimizers and loss functions, each one suiting different applications.

optimizer = torch.optim.Adam(model.parameters(), lr=0.5)  # creating the optimizer
loss_fn = torch.nn.MSELoss()  # creating the MSEloss function from Pytorch
# https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html

# +
# %%time


for iteration in range(500):

    optimizer.zero_grad()

    model.forward()  # get the predicted model
    sky_cube = model.icube.sky_cube

    loss = loss_fn(sky_cube, dirty_image)  # calculates the loss

    loss.backward()  # caulculate gradients of parameters
    optimizer.step()  # updates the parameters
# -

# In this tutorial we will be using different methods of RML optimization so we have to save the model, letting us start from the this clean and better starting point each time. [Information on saving and loading models and the state_dict can be found here.](https://pytorch.org/tutorials/beginner/saving_loading_models.html)

torch.save(model.state_dict(), "model.pt")

# Now we can see the results, the image cube now closely resembles the dirty image (constrained by the fact that it can contain no negative values).

# +

fig, ax = plt.subplots(ncols=2, figsize=(8, 4))

imin, imax = np.amin(img), np.amax(img)

im = ax[0].imshow(
    np.squeeze(dirty_image.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
    vmin=imin,
    vmax=imax,
    # cmap = "Spectral", have these here in case of wanting a more obvious way of seeing negative values
)

im = ax[1].imshow(
    np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
    vmin=imin,
    vmax=imax,
    # cmap = "Spectral", ditto
)

ax[0].set_xlim(left=0.75, right=-0.75)
ax[0].set_ylim(bottom=-0.75, top=0.75)
ax[0].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[0].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
ax[0].set_title("Dirty Image")
ax[1].set_xlim(left=0.75, right=-0.75)
ax[1].set_ylim(bottom=-0.75, top=0.75)
ax[1].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[1].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
ax[1].set_title("Image Cube")
plt.tight_layout()

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.84, 0.17, 0.03, 0.7])
fig.colorbar(im, cax=cbar_ax)


# -

# ## Training Function

# ## Still need to revamp this and the crossvalidation one
#
# Need to make this one using the train function that's under cross validation and then cross validaiton just has the cross validation part. Submitting what I have for rn though. Right now the training function part we had was the initializing part.


def log_figure(model, residuals):
    """
    Create a matplotlib figure showing the current image state.

    Args:
        model: neural net model
    """

    # populate residual connector
    residuals()

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    im = ax[0, 0].imshow(
        np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
        origin="lower",
        interpolation="none",
        extent=model.icube.coords.img_ext,
    )
    plt.colorbar(im, ax=ax[0, 0])

    im = ax[0, 1].imshow(
        np.squeeze(residuals.sky_cube.detach().cpu().numpy()),
        origin="lower",
        interpolation="none",
        extent=residuals.coords.img_ext,
    )
    plt.colorbar(im, ax=ax[0, 1])

    im = ax[1, 0].imshow(
        np.squeeze(torch.log(model.fcube.ground_amp.detach()).cpu().numpy()),
        origin="lower",
        interpolation="none",
        extent=residuals.coords.vis_ext,
    )
    plt.colorbar(im, ax=ax[1, 0])

    im = ax[1, 1].imshow(
        np.squeeze(torch.log(residuals.ground_amp.detach()).cpu().numpy()),
        origin="lower",
        interpolation="none",
        extent=residuals.coords.vis_ext,
    )
    plt.colorbar(im, ax=ax[1, 1])

    return fig


from mpol import losses, connectors

model.load_state_dict(torch.load("model.pt"))
dataset = gridder.to_pytorch_dataset()


def train(model, dataset, optimizer, config, writer=None, report=False, logevery=50):
    model.train()
    residuals = connectors.GriddedResidualConnector(model.fcube, dataset)
    for iteration in range(config["epochs"]):
        optimizer.zero_grad()
        vis = model.forward()
        sky_cube = model.icube.sky_cube
        # computing loss through MPoL loss function with more parameters
        loss = (
            losses.nll_gridded(vis, dataset)
            + config["lambda_sparsity"] * losses.sparsity(sky_cube)
            + config["lambda_TV"] * losses.TV_image(sky_cube)
            + config["entropy"] * losses.entropy(sky_cube, config["prior_intensity"])
        )

        if (iteration % logevery == 0) and writer is not None:
            writer.add_scalar("loss", loss.item(), iteration)
            writer.add_figure("image", log_figure(model, residuals), iteration)

        loss.backward()
        optimizer.step()

    if report:
        tune.report(loss=loss.item())

    return loss.item()


config = {
    "lr": 0.3,
    "lambda_sparsity": 7.076022085822013e-05,
    "lambda_TV": 0.00,
    "entropy": 1e-03,
    "prior_intensity": 1.597766235483388e-07,
    "epochs": 1000,
}

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# +
# %%time

train(model, dataset, optimizer, config, writer=writer, report="True")

# (Edit) Below we can see the loss function, images, and residuals for every saved iteration.
# Be sure that your window is wide enough such that you can navigate to the images tab within tensorboard

# %tensorboard --logdir {logs_base_dir}

# +
# Note- the first image produced below is not necessary as it is included in tensorboard- maybe remove?
fig, ax = plt.subplots(nrows=1, figsize=(8, 8))
im = ax.imshow(
    np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
)
plt.colorbar(im)


def scale(I):
    a = 0.02
    return np.arcsinh(I / a) / np.arcsinh(1 / a)


fig, ax = plt.subplots(nrows=1, figsize=(8, 8))
im = ax.imshow(
    scale(np.squeeze(model.icube.sky_cube.detach().cpu().numpy())),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
)
plt.colorbar(im)

# -

# ### Cross Validation
#
# Now we will move into the realm of Cross Validation. To do this we will be utilizing the [Ray[Tune]](https://docs.ray.io/en/master/tune/index.html) python package for hyperparameter tuning. In order to get the best fit, we will be modifying our `train` function to encorperate a stronger loss function. We will import this from `mpol.losses`. We also need the `mpol.connectors` package because to calculate the residuals. Let us do that now.

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
        model.load_state_dict(MODEL_PATH)

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
    cross_validate(model, config, k_fold_datasets, MODEL_PATH)


from mpol import datasets

dartboard = datasets.Dartboard(coords=coords)
# create cross validator using this "dartboard"
k = 5
cv = datasets.KFoldCrossValidatorGridded(dataset, k, dartboard=dartboard, npseed=42)
k_fold_datasets = [(train, test) for (train, test) in cv]

import ray
from ray import tune

# +
# making sure that we don't initialize ray if its already initialized
ray.shutdown()
ray.init()
MODEL_PATH = "./model.pt"
analysis = tune.run(
    trainable,
    config={
        "lr": 0.3,
        "lambda_sparsity": tune.loguniform(1e-8, 1e-4),
        "lambda_TV": tune.loguniform(1e-4, 1e1),
        "entropy": tune.loguniform(1e-7, 1e-1),
        "prior_intensity": tune.loguniform(1e-8, 1e-4),
        "epochs": 1000,
    },
    num_samples=24,
    resources_per_trial={"cpu": 3},
)

print("Best config: ", analysis.get_best_config(metric="cv_score", mode="min"))
# -

# Get a dataframe for analyzing trial results.
df = analysis.results_df
print(df)

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
ax[0].set_title("Dirty Image")
ax[1].set_xlim(left=0.75, right=-0.75)
ax[1].set_ylim(bottom=-0.75, top=0.75)
ax[1].set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax[1].set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
ax[1].set_title("Optimized Image")
plt.tight_layout()
# -
