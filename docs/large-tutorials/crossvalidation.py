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

# ## Cross validation
#
# In this tutorial, we'll design an imaging workflow that will help us build confidence that we are setting the regularization hyperparameters appropriately.

# # Setup
#
# We'll continue with the same ALMA logo measurement set as before. If these commands don't make sense, please consult the previous tutorials.

# +
import torch
import numpy as np
import matplotlib.pyplot as plt
import mpol
from mpol import gridding, coordinates, precomposed, losses, images
import requests
from astropy.utils.data import download_file
import os

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
coords = gridding.GridCoords(cell_size=0.005, npix=800)
gridder = gridding.Gridder(
    coords=coords, uu=uu, vv=vv, weight=weight, data_re=data_re, data_im=data_im,
)

# export to PyTorch dataset
gridder.grid_visibilities(weighting="uniform")
dset = gridder.to_pytorch_dataset()
# -

# ### K-fold cross validation
# The big question here is what to use as a training set and what to use as a test set?
# One approach would be to just choose at random 1/10th of the loose visibilities, or 1/10th of the gridded visibilities.
# We conjecture that the more important aspect of interferometric observations with arrays like ALMA or JVLA are the unsampled visibilities that carry significant power. A scheme like the one described would not simulate the "holes" in the uv coverage, but because the individual visibilities are so dense, that randomly selecting and dropping them out wouldn't simulate the missing data from the observation.

# Instead, we suggest an approach where we break the UV plane into radial ($q=\sqrt{u^2 + v^2}$) and azimuthal ($\phi = \mathrm{arctan2}(v,u)$) cells . There are, of course, no limits on how you choose to cross-validate your datasets; there are most likely other methods that will work well depending on the dataset.

# Visualize the grid itself. Make a plot of the polar cells and locations in both linear and log space.

# Visualize the gridded locations (non-zero histogram)

# Visualize the process of choosing different subsets of gridded locations
# have the original gridded locations in one pale color, non cell locations in white, and the chosen ones in red or something

# show the dirty images corresponding to each selected dartboard.
# ResidualConnector between zeroed FourierLayer and Dataset will make a dirty image.

# Design a cross validation training loop, reporting the key metric of cross validated score.
# Make sure we can iterate through the same datasets (keeping random seed).

# Restart the training loop idea, first using only chi^2 to get CV score benchmark

# Then try with sparsity, and try out a low and medium value to see if CV score improves, hopefully landing somewhere at a minimum.


# +
# create a partition
dartboard = datasets.Dartboard(coords=coords)

# create cross validator through passing dartboard
k = 5
cv = datasets.KFoldCrossValidatorGridded(dataset, k, dartboard=dartboard)
# -

flayer = images.FourierCube(coords=coords)
flayer.forward(torch.zeros(dataset.nchan, coords.npix, coords.npix))

# We then initialize SimpleNet with the relevant information
k_fold_datasets = [(train, test) for item in cv]

for k, (train, test) in enumerate(k_fold_datasets):

        rtrain = connectors.GriddedResidualConnector(flayer, train)
        rtest = connectors.GriddedResidualConnector(flayer, test)

        train_chan = images.packed_cube_to_sky_cube(rtrain.forward())[chan]
        test_chan = images.packed_cube_to_sky_cube(rtest.forward())[chan]

        im = ax[k, 0].imshow(
            train_chan.real.detach().numpy(), interpolation="none", origin="lower"
        )
        plt.colorbar(im, ax=ax[k, 0])

        im = ax[k, 1].imshow(
            train_chan.imag.detach().numpy(), interpolation="none", origin="lower"
        )
        plt.colorbar(im, ax=ax[k, 1])

        im = ax[k, 2].imshow(
            test_chan.real.detach().numpy(), interpolation="none", origin="lower"
        )
        plt.colorbar(im, ax=ax[k, 2])

        im = ax[k, 3].imshow(
            test_chan.imag.detach().numpy(), interpolation="none", origin="lower"
        )
        plt.colorbar(im, ax=ax[k, 3])


# We want to test a loss function of the form
#
# $$
# f_\mathrm{loss} = f_\mathrm{nll} + \lambda_\mathrm{sparsity} f_\mathrm{sparsity} + \lambda_{TSV} f_\mathrm{TSV}
# $$

def cross_validate(lambda_sparsity, lambda_tsv, train_epochs=300):
    """
    loss_fn takes in model_visibilities, data_visibilities, 
    """
    
    test_scores = []
    afor k, (train, test) in enumerate(k_fold_datasets):
        rml_train = precomposed.SimpleNet(coords=coords, nchan=train.nchan, griddedDataset=train)
        rml_test = precomposed.SimpleNet(coords=coords, nchan=test.nchan, griddedDataset=test)

        for i in range(train_epochs):
            rml_train.zero_grad()

            # get the predicted model
            model_visibilities = rml_train.forward()

            # calculate a loss
            loss = losses.nll(model_visibilities, train.vis_indexed, train.weight_indexed) \
                + lambda_sparsity * losses.
            
            # calc sparsity
            # calc tsv

            loss_tracker.append(loss.item())

            # calculate gradients of parameters
            loss.backward()

            # update the model parameters
            optimizer.step()

        # evaluate test score 
        test_visibilities = rml_test.forward()
        test_scores.append(losses.nll(test_visibilities, test.vis_indexed, test.weight_indexed))


    # aggregate all test scores and sum to evaluate cross val metric
    test_score = np.sum(np.array(test_scores))
    return test_score

# +
# now calculate test_score for as many settings of sparsity and TSV as you want, starting from 0 at each. 
# Assuming that the iterations acutally converge.
