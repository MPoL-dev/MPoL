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

# # HD143006 Tutorial Part 2
#
# This tutorial is a continuation of the [HD143006 Part 1](HD143006_Part_1.html) tutorial. It covers the same content as the MPoL tutorials on [Optimization](https://mpol-dev.github.io/MPoL/ci-tutorials/optimization.html), [Initalizing with the Dirty Image](https://mpol-dev.github.io/MPoL/ci-tutorials/initializedirtyimage.html), and [Cross Validation](https://mpol-dev.github.io/MPoL/ci-tutorials/crossvalidation.html) but in a streamlined fashion and using real data. These other tutorials provide a more comprehensive breakdown of each step in this tutorial.
#
# This tutorial will be going through how to initialize the model, the imaging and optimization process, how to use cross validation to improve the choice of hyperparameters in the model to more accurately predict new data, and how to analyze the results of our work with TensorBoard.
#
# ### Loading Data
# Let's load the data as we did in the previous HD143006 tutorial ([Part 1](HD143006_Part_1.html)) and create the MPoL Gridder object.
#
# *You can either download these two files (HD143006_continuum.fits and HD143006_continuum.npz) directly to your working directory, or use Astropy to download them during run time.*

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
    data_re=data.real,  # separating the real and imaginary values of our data
    data_im=data.imag,
)

# -

#
# ### Getting the Dirty Image and Creating the Model
#
# First, we are going to get the dirty image from our Gridder object. We will use the Briggs weighting scale and set `robust=0.0` here as this option leads to a dirty image resembling the DSHARP CLEAN image (see [Part 1](HD143006_Part_1.html)).

img, beam = gridder.get_dirty_image(weighting="briggs", robust=0.0, unit="Jy/arcsec^2")

# We now have everything from the last tutorial loaded and we can create the model.

from mpol.precomposed import SimpleNet

model = SimpleNet(coords=coords, nchan=gridder.nchan)

import torch

# taking the dirty image and making it a tensor
dirty_image = torch.tensor(img.copy())

# ## Initializing Model with the Dirty Image
#
# We now have our model and data, but before we set out trying to optimize the image we need to choose a starting point for our future optimization loops. This starting point could be anything, but a good choice is the dirty image, since it is already a maximum likelihood fit to the data. The problem with this is that the dirty image contains noise and negative flux pixels, while we seek to limit noise and impose the requirement that our sources must have all positive flux values. Our solution then is to train the RML model so that its parameters more closely resemble the dirty image while enforcing the positive flux prior .
#


# To optimize the RML model toward the dirty image, we will create our training loop using a [loss function](.https://mpol-dev.github.io/MPoL/api.html#module-mpol.losses) and an [optimizer](https://pytorch.org/docs/stable/optim.html#module-torch.optim). This process is described in greater detail in the [Optimization Loop](https://mpol-dev.github.io/MPoL/ci-tutorials/optimization.html) and [Initalizing with the Dirty Image](https://mpol-dev.github.io/MPoL/ci-tutorials/initializedirtyimage.html) tutorials. MPoL and PyTorch both contain many optimizers and loss functions, each one suiting different applications. Here we use PyTorch's [mean squared error function](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html) between the RML model image pixel fluxes and the dirty image pixel fluxes.

optimizer = torch.optim.Adam(model.parameters(), lr=0.5)  # creating the optimizer
loss_fn = torch.nn.MSELoss()  # creating the MSEloss function from Pytorch

# +
# %%time


for iteration in range(500):

    optimizer.zero_grad()

    model.forward()  # get the predicted model
    sky_cube = model.icube.sky_cube

    loss = loss_fn(sky_cube, dirty_image)  # calculate the loss

    loss.backward()  # calculate gradients of parameters
    optimizer.step()  # update the parameters
# -

# In this tutorial we will be running through RML optimization multiple times with different configurations of hyperparameters and then comparing them so we have to save the model, letting us start from this clean starting point each time. Information on saving and loading models and the state_dict can be found [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html).

torch.save(model.state_dict(), "model.pt")

# Now we can see the results, the image cube now closely resembles the dirty image (constrained by the fact that it can contain no negative flux pixel values).

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
)

im = ax[1].imshow(
    np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
    vmin=imin,
    vmax=imax,
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
plt.tight_layout()


# -

# ## Training and Imaging Part 1

# Now that we have a better starting point, we can work on optimizing our image using a more complex training function that has additional priors and regularizers. This part of the tutorial will also use [TensorBoard](https://pytorch.org/docs/stable/tensorboard.html) to display the loss function and changes in the image through each saved iteration of the training loop. This will allow us to better determine the hyperparameters to be used (a hyperparameter is a parameter of the model set by the user to control the learning process and can not be predicted by the model). Note that to display the TensorBoard dashboards referenced in this tutorial, you will need to run these commands on your own device. The code necessary to do this is displayed in this tutorial as ``#%tensorboard --logdir <directory>``. Uncomment and execute this IPython line magic command in Jupyter Notebook to open the dashboard.


# Here we set up the tools that will allow us to visualize the results of the loop in TensorBoard.

from mpol import (
    losses,  # here MPoL loss functions will be used
    connectors,  # required to calculate the residuals in log_figure
)


# setting up Writer to log values and images for display in TensorBoard
from torch.utils.tensorboard import SummaryWriter
import os


logs_base_dir = "./logs/"
writer = SummaryWriter(logs_base_dir)
os.makedirs(logs_base_dir, exist_ok=True)
# %load_ext tensorboard


def log_figure(
    model, residuals
):  # this function takes a snapshot of the image state, imaged residuals, amplitude of model visibilities, and amplitude of residuals

    # populate residual connector
    residuals()

    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
    im = ax[0, 0].imshow(
        np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),  #
        origin="lower",
        interpolation="none",
        extent=model.icube.coords.img_ext,
    )
    plt.colorbar(im, ax=ax[0, 0])

    im = ax[0, 1].imshow(
        np.squeeze(residuals.sky_cube.detach().cpu().numpy()),  #
        origin="lower",
        interpolation="none",
        extent=residuals.coords.img_ext,
    )
    plt.colorbar(im, ax=ax[0, 1])

    im = ax[1, 0].imshow(
        np.squeeze(torch.log(model.fcube.ground_amp.detach()).cpu().numpy()),  #
        origin="lower",
        interpolation="none",
        extent=residuals.coords.vis_ext,
    )
    plt.colorbar(im, ax=ax[1, 0])

    im = ax[1, 1].imshow(
        np.squeeze(torch.log(residuals.ground_amp.detach()).cpu().numpy()),  #
        origin="lower",
        interpolation="none",
        extent=residuals.coords.vis_ext,
    )
    plt.colorbar(im, ax=ax[1, 1])

    return fig


# With these set up, we can now make our training function (instead of a loop, we use a function here since the training loop will be ran multiple times with different configurations, the configurations consisting of changed hyperparameter values). Our new training function includes additional priors and loss functions, this helps to further guide our model's learning in a direction based on some assumptions we make. To learn more information about these, please see the [Losses API](https://mpol-dev.github.io/MPoL/api.html#module-mpol.losses) and the [Introduction to Regularized Maxium Likelihood Imaging page](https://mpol-dev.github.io/MPoL/rml_intro.html). Later in the tutorial, we will see how these can affect the resulting image.


def train(model, dataset, optimizer, config, writer=None, logevery=50):
    model.train()
    residuals = connectors.GriddedResidualConnector(model.fcube, dataset)
    for iteration in range(config["epochs"]):
        optimizer.zero_grad()
        vis = model.forward()  # get the predicted model
        sky_cube = model.icube.sky_cube
        # computing loss through MPoL loss function with more parameters
        loss = (
            losses.nll_gridded(vis, dataset)
            + config["lambda_sparsity"] * losses.sparsity(sky_cube)
            + config["lambda_TV"] * losses.TV_image(sky_cube)
            + config["entropy"] * losses.entropy(sky_cube, config["prior_intensity"])
        )

        if (
            iteration % logevery == 0
        ) and writer is not None:  # logging the loss and image for visualization and analysis
            writer.add_scalar("loss", loss.item(), iteration)
            writer.add_figure("image", log_figure(model, residuals), iteration)

        loss.backward()  # calculate gradient of the parameters
        optimizer.step()  # update the model parameters

    return loss.item()


# With our function created, all that is left is to load the initialized model, export the visibilities to a PyTorch dataset, set our hyperparameters, and create our optimizer.

# +
model.load_state_dict(
    torch.load("model.pt")
)  # load our initialized model from the previous section

dataset = (
    gridder.to_pytorch_dataset()
)  # export the visibilities from gridder to a PyTorch dataset
# -

# Here is where we define the hyperparameters under the `config` dictionary. The hyperparameters (also referred to as scalar prefactors in the [Introduction to Regularized Maxium Likelihood Imaging page](https://mpol-dev.github.io/MPoL/rml_intro.html). Most of these hyperparameters, such as `lambda_TV` and `entropy` are used in the loss functions and can be read about [here](https://mpol-dev.github.io/MPoL/api.html#module-mpol.losses).

config = (
    {  # config includes the hyperparameters used in the function and in the optimizer
        "lr": 0.3,
        "lambda_sparsity": 7.076022085822013e-05,
        "lambda_TV": 0.00,
        "entropy": 1e-03,
        "prior_intensity": 1.597766235483388e-07,
        "epochs": 1000,
    }
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=config["lr"]
)  # create our optimizer, using the learning rate from config

# We are now ready to run the training loop.

# +
# %%time

train(model, dataset, optimizer, config, writer=writer)
# -

# Below we can see the loss function, images, and residuals for every saved iteration including our final result. To view the loss function, navigate to the scalars tab. To view the four images, be sure your window is wide enough to navigate to the images tab within TensorBoard. The images, in order from left-right top-bottom are: image cube representation, imaged residuals, visibility amplitudes of model on a log scale, residual amplitudes on a log scale. You can use the slider to view different iterations.

# +
# # %tensorboard --logdir {logs_base_dir}
## uncomment the above line when running to view TensorBoard
# -

# ## Training and Imaging Part 2: Cross Validation
#
# Now we will move into the realm of cross validation. Cross validation is a technique used to assess model validity. This is completed by storing one chunk of a dataset as the test dataset and using the remaining data to train the model. Once the model is trained, it is used to predict the values of the data in the test dataset. These predicted values are compared to the values from the test dataset, producing a cross validation score. The advantage of cross validation is that it allows one dataset to be used to train the model multiple times since it can take different chunks out for the test dataset. For more information see the [Cross Validation tutorial](https://mpol-dev.github.io/MPoL/ci-tutorials/crossvalidation.html).
#
# Just like in the previous section we will be viewing our results in TensorBoard, with the addition of the cross validation score log.

# Cross validation requires a `test` function (to determine the cross validation score) and a `cross_validate` function (to utilize cross validation with the previous `train` function). We implement these below.


def test(model, dataset):
    model.eval()
    vis = model.forward()
    loss = losses.nll_gridded(
        vis, dataset
    )  # calculates the loss function that goes to make up the cross validation score
    return loss.item()


def cross_validate(model, config, k_fold_datasets, MODEL_PATH, writer=None):
    test_scores = []

    for k_fold, (train_dset, test_dset) in enumerate(k_fold_datasets):

        # reset model
        model.load_state_dict(torch.load(MODEL_PATH))

        # create a new optimizer for this k_fold
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

        # train for a while
        train(model, train_dset, optimizer, config, writer=writer)
        # evaluate the test metric
        test_scores.append(test(model, test_dset))

    # aggregate all test scores and sum to evaluate cross val metric
    test_score = np.sum(np.array(test_scores))

    # adds cross validation score
    if writer is not None:
        writer.add_scalar("Cross Validation", test_score)

    return test_score


# Now, with our functions defined, we need to do the critical part of dividing our dataset into training and test datasets. There are many ways of going about this but here we are splitting it radially and azimuthally and removing chunks. MPoL's `Dartboard` presents an easy built-in way to get the polar coordinate grid of a dataset. To to read about why and how this works, and to visualize this, please see [Choosing the K-folds](https://mpol-dev.github.io/MPoL/ci-tutorials/crossvalidation.html#Choosing-the-K-folds) in the [Cross Validation tutorial](https://mpol-dev.github.io/MPoL/ci-tutorials/crossvalidation.html).

from mpol import datasets

# +
# create a radial and azimuthal partition for the dataset
dartboard = datasets.Dartboard(coords=coords)

# create cross validator using this "dartboard"
k = 5
cv = datasets.KFoldCrossValidatorGridded(dataset, k, dartboard=dartboard, npseed=42)

# ``cv`` is a Python iterator, it will return a ``(train, test)`` pair of ``GriddedDataset``s for each iteration.
# Because we'll want to revisit the individual datasets
# several times in this tutorial, we're storing them into a list

k_fold_datasets = [(train, test) for (train, test) in cv]
# -


# If you recall, we saved the trained model's state. Here we will be utilizing this. `MODEL_PATH` is defined below so we can reset the model between cross validation loops by reloading `model.pt`. We will run the cross validation loops for a few different configurations, starting with the hyperparameters found in `config`, defined above in *Training and Imaging Part 1* of this tutorial. This configuration has been included in the following cell for convenience.

# +
MODEL_PATH = "model.pt"

new_config = (
    {  # config includes the hyperparameters used in the function and in the optimizer
        "lr": 0.3,
        "lambda_sparsity": 7.076022085822013e-05,
        "lambda_TV": 0.00,
        "entropy": 1e-03,
        "prior_intensity": 1.597766235483388e-07,
        "epochs": 1000,
    }
)
# -


# We are now ready to run our cross validation loop. We'll run this a few times while changing hyperparameters in the config to lower the cross validation score then compare all three with TensorBoard.

# +
# %%time

# new directory to write the progress of our first cross val. loop to
cv_log_dir1 = logs_base_dir + "cv/cv1/"
cv_writer1 = SummaryWriter(cv_log_dir1)
os.makedirs(cv_log_dir1, exist_ok=True)

cv_score1 = cross_validate(
    model, new_config, k_fold_datasets, MODEL_PATH, writer=cv_writer1
)
print(f"Cross Validation Score: {cv_score1}")

# +
# %%time

# new directory to write the progress of our second cross val. loop to
cv_log_dir2 = logs_base_dir + "cv/cv2/"
cv_writer2 = SummaryWriter(cv_log_dir2)
os.makedirs(cv_log_dir2, exist_ok=True)

new_config = (
    {  # config includes the hyperparameters used in the function and in the optimizer
        "lr": 0.3,
        "lambda_sparsity": 1.0e-4,
        "lambda_TV": 1.0e-4,
        "entropy": 1e-02,
        "prior_intensity": 2.0e-09,
        "epochs": 850,
    }
)
cv_score2 = cross_validate(
    model, new_config, k_fold_datasets, MODEL_PATH, writer=cv_writer2
)
print(f"Cross Validation Score: {cv_score2}")


# +
# %%time

# new directory to write the progress of our third cross val. loop to
cv_log_dir3 = logs_base_dir + "cv/cv3/"
cv_writer3 = SummaryWriter(cv_log_dir3)
os.makedirs(cv_log_dir3, exist_ok=True)

new_config = (
    {  # config includes the hyperparameters used in the function and in the optimizer
        "lr": 0.3,
        "lambda_sparsity": 1.0e-3,
        "lambda_TV": 1.2e-4,
        "entropy": 1e-02,
        "prior_intensity": 2.0e-09,
        "epochs": 400,
    }
)

cv_score3 = cross_validate(
    model, new_config, k_fold_datasets, MODEL_PATH, writer=cv_writer3
)
print(f"Cross Validation Score: {cv_score3}")
# -

# Here is the final result for our model with the lowest cross validation score:

fig, ax = plt.subplots()
im = ax.imshow(
    np.squeeze(model.icube.sky_cube.detach().cpu().numpy()),
    origin="lower",
    interpolation="none",
    extent=model.icube.coords.img_ext,
)
plt.colorbar(im)

# Below are the results in the TensorBoard. Note that while it may seem strange that the lowest loss values do not correspond with the lowest cross validation scores, different hyperparameters in the config dictionaries increase the weight of some loss functions (and others, such as hyperparameters set to zero, remove some loss function components completely) so the loss values are not perfectly equal representations across each configuration and run.

cv_log_dir = logs_base_dir + "cv/"
# # %tensorboard --logdir {cv_log_dir}
## uncomment the above line when running to view TensorBoard

# Now with this tutorial done we can see the results of RML imaging; an image optimized to fit the provided dataset. By initializing the model with the dirty image we are able to have our model converge to the optimal image in fewer iterations. We were also able to arrive at a more statistically accurate image by using cross validation. From the TensorBoard, we are able to see how changing hyperparameters can result in a lower cross validation score, and therefore a better image, if done correctly. This process of changing the hyperparameters can be automated using Ray Tune, as we will explore in Part 3 of this tutorial series. Of the three configurations we've displayed above, the third has the lowest cross validation score. When we compare the final image of each of these three configurations, we see the third image is most similar to the image produced using the CLEAN algorithm and is an improvement from the dirty image we obtained in Part 1 of this tutorial series. The third image is less sparse than the first image, and it is less noisy than the second image and dirty image. If you would like to compare these results yourself, please run TensorBoard locally. In the next part of the HD143006 tutorial we will be expanding on how to analyze the results of the training, optimization loops, hyperparameter tuning, and exploring the full pipeline of data analysis which can be adapted to any real world data.
