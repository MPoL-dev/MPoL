# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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
# %run notebook_setup
# -

# ## Cross validation
#
# In this tutorial, we'll design and optimize a more sophisticated imaging workflow. Cross validation will help us build confidence that we are setting the regularization hyperparameters appropriately.
#
# # Setup
#
# We'll continue with the same central channel of the ALMA logo measurement set as before. If these commands don't make sense, please consult the previous tutorials.

# +
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpol import (
    gridding,
    precomposed,
    losses,
    images,
    datasets,
    connectors,
)
from astropy.utils.data import download_file
from torch.utils.tensorboard import SummaryWriter

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

# define the image dimensions, as in the previous tutorial
coords = gridding.GridCoords(cell_size=0.04, npix=128)
gridder = gridding.Gridder(
    coords=coords, uu=uu, vv=vv, weight=weight, data_re=data_re, data_im=data_im,
)

# export to PyTorch dataset
gridder.grid_visibilities(weighting="uniform")
dset = gridder.to_pytorch_dataset()
# -

# ### K-fold cross validation
#
# [K-fold cross validation](https://en.wikipedia.org/wiki/Cross-validation_(statistics)) is a technique used to assess model validity. In the context of RML imaging, we use "model" to describe a whole host of assumptions inherent to the imaging workflow. Model settings include the ``cell_size``, the number of pixels, the mapping of the BaseCube to the ImageCube, as well as hyperparameter choices like the strength of the regularizer terms for each type of loss function. Usually we're most interested in assessing whether we have adequately set hyperparameters (like in this tutorial), but sometimes we'd like to assess model settings too.
#
# If you're coming from astronomy or astrophysics, you might be most familiar with doing [Bayesian parameter inference](https://ui.adsabs.harvard.edu/abs/2019arXiv190912313S/abstract) with all of the data at once. In a typical workflow, you might implicitly assume that your model is valid and explore the shape of the unnormalized posterior distribution using a standard MCMC technique like Metropolis-Hastings. If you did want to compare the validity of models, then you would need to use a sampler which computes the Bayesian evidence, or posterior normalization.
#
# But if you're coming from the machine learning community, you're most likely already familiar from the concept of optimizing your model using a "training" dataset and the assessing how well it does using a "test" or "validation" dataset. Astrophysical datasets are typically hard-won, however, so it's not often that we have a sizeable chunk of data lying around to use as a test set *in addition to* what we want to incorporate into our training dataset.
#
# $K$-fold cross validation helps alleviate this concern somewhat by rotating testing/training chunks through the dataset. To implement $K$-fold cross validation, first split your dataset into $K$ (approximately equal) chunks. Then, do the following $K$ times:

# * store one chunk ($1/K$th of the total data) separately as a test dataset
# * combine the remaining chunks ($(K-1)/K$ of the total data set) into one dataset and use this to train the model
# * use this model to predict the values of the data in the test dataset
# * assess the difference between predicted test data and actual test data using a $\chi^2$ metric, called the cross-validation score

# When all loops are done, you can average the $K$ cross-validation scores together into a final score for that model configuration. Lower cross validation scores are better in the sense that the trained model did a better job predicting the test data.
#
# **Why does this work?** Cross validation is such a useful tool because it tells us how well a model generalizes to new data, with the idea being that a better model will predict new data more accurately. Some more considered thoughts on cross validation and model fitting are in [Hogg and Villar](https://ui.adsabs.harvard.edu/abs/2021arXiv210107256H/abstract).

# ### Choosing the $K$-folds
#
# There are many ways to split a dataset into $K$ chunks, and, depending on your application, some schemes are better than others. For most interferometric datasets, visibility samples are clustered in Fourier space due to the limitations on the number and location of the antennas. One objective of cross validation might be figuring out how sparse $u$,$v$ coverage adversely affects our imaging process---ideally we'd like to tune the algorithm such that we would still recover a similar image even if our $u$,$v$ sampling were different. To explore slicing choices, here is the full $u$,$v$ coverage of our ALMA logo mock dataset (C43-7, 1 hour observation)

fig, ax = plt.subplots(nrows=1)
ax.scatter(uu, vv, s=1.5, rasterized=True, linewidths=0.0, c="k")
ax.scatter(
    -uu, -vv, s=1.5, rasterized=True, linewidths=0.0, c="k"
)  # and Hermitian conjugates
ax.set_xlabel(r"$u$ [k$\lambda$]")
ax.set_ylabel(r"$v$ [k$\lambda$]")
ax.set_title("original dataset")
ax.invert_xaxis()

# As you can see, the $u$,$v$ space is sampled in a very structured way:
#
# 1. there are no samples at very low spatial frequencies (the center of the image, $< 10$ k$\lambda$)
# 2. most samples like at intermediate spatial frequencies (100 k$\lambda$ to 800 k$\lambda$)
# 3. there are very few samples at high spatial frequencies ($>$ 1000 k$\lambda$)
# 4. there are many gaps in the $u$,$v$ coverage at high spatial frequencies
#
# If we were to just randomly draw chunks from these visibilities, because there are so many visibilities, we would end up mostly replicating the same structured pattern. For example, here is what random training set would look like for $K=10$

# +
nvis = len(uu)
ind = np.random.choice(np.arange(nvis), size=int(9 * nvis / 10), replace=False)

uk = uu[ind]
vk = vv[ind]

fig, ax = plt.subplots(nrows=1)
ax.scatter(uk, vk, s=1.5, rasterized=True, linewidths=0.0, c="k")
ax.scatter(
    -uk, -vk, s=1.5, rasterized=True, linewidths=0.0, c="k"
)  # and Hermitian conjugates
ax.set_xlabel(r"$u$ [k$\lambda$]")
ax.set_ylabel(r"$v$ [k$\lambda$]")
ax.set_title("randomly drawn 9/10 dataset")
ax.invert_xaxis()
# -

# As you can see, this training set looks very similar to the full dataset, with the same holes in $u$,$v$ coverage and the same sampling densities. So, randomly generated training datasets don't really stress test the model in any new or interesting ways relative to the full dataset.
#
# But, the missing holes in the real dataset are quite important to image fidelity---if we had complete $u$,$v$ coverage, we wouldn't need to be worrying about CLEAN or RML imaging techniques in the first place. When we make a new interferometric observation, it will have it's own (different) set of missing holes depending on array configuration, observation duration, and hour angle coverage. We would like our cross validation slices to simulate the distribution of possible new datasets, and, at least for ALMA, random sampling doesn't accomplish this.
#
# Instead, we suggest an approach where we break the UV plane into radial ($q=\sqrt{u^2 + v^2}$) and azimuthal ($\phi = \mathrm{arctan2}(v,u)$) cells and cross validate by drawing a subselection of these cells. There are, of course, no limits on how you might split your dataset for cross-validation; it really depends on what works best for your goals.

# +
# create a partition
dartboard = datasets.Dartboard(coords=coords)

# create cross validator through passing dartboard
k = 5
cv = datasets.KFoldCrossValidatorGridded(dset, k, dartboard=dartboard, npseed=42)

# store output into a list for now, since we'll use it a bunch
k_fold_datasets = [(train, test) for (train, test) in cv]
# -

# Visualize the grid itself. Make a plot of the polar cells and locations in both linear and log space.

flayer = images.FourierCube(coords=coords)
flayer.forward(torch.zeros(dset.nchan, coords.npix, coords.npix))

# Visualize the gridded locations (non-zero histogram). UV in $\mathrm{k}\lambda$ and image is in arcseconds.

# +
fig, ax = plt.subplots(nrows=k, ncols=3, figsize=(6, 10))

for i, (train, test) in enumerate(k_fold_datasets):

    rtrain = connectors.GriddedResidualConnector(flayer, train)
    rtrain.forward()
    rtest = connectors.GriddedResidualConnector(flayer, test)
    rtest.forward()

    vis_ext = rtrain.coords.vis_ext
    img_ext = rtrain.coords.img_ext

    train_mask = rtrain.ground_mask[0]
    train_chan = rtrain.sky_cube[0]

    test_mask = rtest.ground_mask[0]
    test_chan = rtest.sky_cube[0]

    ax[i, 0].imshow(
        train_mask.detach().numpy(),
        interpolation="none",
        origin="lower",
        extent=vis_ext,
        cmap="GnBu",
    )

    ax[i, 1].imshow(
        train_chan.detach().numpy(),
        interpolation="none",
        origin="lower",
        extent=img_ext,
    )

    ax[i, 2].imshow(
        test_mask.detach().numpy(),
        interpolation="none",
        origin="lower",
        extent=vis_ext,
        cmap="GnBu",
    )

    ax[i, 0].set_ylabel("k-fold {:}".format(i))

ax[0, 0].set_title("train mask")
ax[0, 1].set_title("train dirty img.")
ax[0, 2].set_title("test mask")

for a in ax.flatten():
    a.xaxis.set_ticklabels([])
    a.yaxis.set_ticklabels([])

fig.subplots_adjust(left=0.15, hspace=0.0, wspace=0.2)
# -

# Following the previous optimization tutorial, let's create a training function.


def train(model, dset, config, optimizer):
    model.train()  # set to training mode

    for i in range(config["epochs"]):
        model.zero_grad()

        # get the predicted model
        vis = model.forward()

        # get the sky cube too
        sky_cube = model.icube.sky_cube

        # calculate a loss
        loss = (
            losses.nll_gridded(vis, dset)
            + config["lambda_sparsity"] * losses.sparsity(sky_cube)
            + config["lambda_TV"] * losses.TV_image(sky_cube)
        )

        # writer.add_scalar("loss", loss.item(), i)

        # calculate gradients of parameters
        loss.backward()

        # update the model parameters
        optimizer.step()


def test(model, dset):
    model.train(False)
    # evaluate test score
    vis = model.forward()
    loss = losses.nll_gridded(vis, dset)
    return loss.item()


# Design a cross validation training loop, reporting the key metric of cross validated score.
# Make sure we can iterate through the same datasets (keeping random seed).

# Restart the training loop idea, first using only chi^2 to get CV score benchmark

# Then try with sparsity, and try out a low and medium value to see if CV score improves, hopefully landing somewhere at a minimum.


# We want to test a loss function of the form
#
# $$
# f_\mathrm{loss} = f_\mathrm{nll} + \lambda_\mathrm{sparsity} f_\mathrm{sparsity} + \lambda_{TSV} f_\mathrm{TSV}
# $$


def cross_validate(config):
    """
    config is a dictionary that should contain ``lr``, ``lambda_sparsity``, ``lambda_TV``, ``epochs``
    """
    test_scores = []

    for k_fold, (train_dset, test_dset) in enumerate(k_fold_datasets):

        # create a new model and optimizer for this k_fold
        rml = precomposed.SimpleNet(coords=coords, nchan=train_dset.nchan)
        optimizer = torch.optim.Adam(rml.parameters(), lr=config["lr"])

        # train for a while
        train(rml, train_dset, config, optimizer)
        # evaluate the test metric
        test_scores.append(test(rml, test_dset))

    # aggregate all test scores and sum to evaluate cross val metric
    test_score = np.sum(np.array(test_scores))

    return test_score


# nll_hparams = {"lr": 500, "lambda_sparsity": 0, "lambda_TV": 0, "epochs": 400}

# print(cross_validate(nll_hparams))
pars = {"lr": 1.0, "lambda_sparsity": 1e-3, "lambda_TV": 1e-4, "epochs": 1000}
cross_validate(pars)


# +
# now calculate test_score for as many settings of sparsity and TSV as you want, starting from 0 at each.
# Assuming that the iterations actually converge.
# -

# train a full model and see what it looks like
rml = precomposed.SimpleNet(coords=coords, nchan=dset.nchan)
optimizer = torch.optim.Adam(rml.parameters(), lr=pars["lr"])
train(rml, dset, pars, optimizer)

img_ext = rml.coords.img_ext
fig, ax = plt.subplots()
ax.imshow(
    np.squeeze(rml.icube.sky_cube.detach().numpy()),
    interpolation="none",
    origin="lower",
    extent=img_ext,
)

