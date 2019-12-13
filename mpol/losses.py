"""
A library of potential loss functions to use in imaging. 

Many of the definitions follow those in Appendix A of EHT-IV 2019: https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract

including the regularization strength, \zeta, which seeks to be of order unity for most applications. This provides at least a useful starting point when starting to tune multiple loss functions.
"""

import numpy as np
import torch

from mpol.constants import *


def loss_fn(model_vis, data_vis):
    """
    Calculate the weighted chi^2 loss between data and model visibilities.

    Args:
        model_vis: 2-tuple of (real, imaginary) values of the model 
        data_vis: the UVDataSet to calculate the loss against

    Returns:
        double
    """
    model_re, model_im = model_vis
    data_re, data_im, data_weights = data_vis
    return torch.sum(data_weights * (data_re - model_re) ** 2) + torch.sum(
        data_weights * (data_im - model_im) ** 2
    )


def loss_fn_entropy(image, prior_intensity):
    """
    Calculate the entropy loss of a set of pixels. Following the entropy definition in Carcamo et al. 2018: https://ui.adsabs.harvard.edu/abs/2018A%26C....22...16C/abstract

    Args:
        image (any tensor): the array and pixel values. Pixel values must be positive.
        prior_intensity (any tensor): the prior value to calculate entropy against. Could be a single constant or an array the same shape as image.

    Returns:
        float : image entropy
    """
    # check to make sure image is positive, otherwise raise an error
    assert (image > 0).all(), "image contained negative pixel values"
    assert prior_intensity > 0, "image prior intensity must be positive"

    norm = image / prior_intensity
    return torch.sum(norm * torch.log(norm))


def loss_fn_TV(image, epsilon=1e-10):
    """
    Calculate the total variation loss. Following the definition in EHT-IV 2019: https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract Promotes the image to be piecewise smooth.

    Args:
        image (any 2D tensor): the array and pixel values
        epsilon (float): the softening parameter in Jy/arcsec^2. Any pixel-to-pixel variations smaller than this parameter will not be penalized.

    Returns:
        float: loss due to total variation
    """
    # calculate the difference of the image with its eastern pixel
    diff_ll = image[:, 1:] - image[:, 0:-1]
    # calculate the difference of the image with its southern pixel
    diff_mm = image[1:, :] - image[0:-1, :]

    # print(diff_ll.shape)
    # print(diff_mm.shape)

    loss = torch.sum(torch.sqrt(diff_ll ** 2 + diff_mm.T ** 2 + epsilon))

    return loss
