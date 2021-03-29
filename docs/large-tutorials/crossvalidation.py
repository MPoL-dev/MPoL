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

