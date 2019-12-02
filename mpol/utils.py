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
    Calculate the entropy loss of an array. Following the entropy definition in EHT-IV-2019: https://ui.adsabs.harvard.edu/abs/2019ApJ...875L...4E/abstract

    Args:
        image (any tensor): the array and pixel values. Pixel values must be positive.
        prior_intensity (any tensor): the prior value to calculate entropy against. Could be a single constant or an array the same shape as image.

    Returns:
        float : image entropy
    """
    # check to make sure image is positive, otherwise raise an error
    assert (image > 0).all(), "image contained negative pixel values"
    assert prior_intensity > 0, "image prior must be positive"

    norm = image / prior_intensity
    return torch.sum(norm * torch.log(norm))


def get_Jy_arcsec2(T_b, nu=230e9):
    """
    Get specific intensity from the brightness temperature.

    Args:
        T_b : brightness temperature in Kelvin
        nu : frequency (in Hz)

    Returns:
        specific intensity (in Jy/arcsec^2)
    """

    # brightness temperature assuming RJ limit
    # units of ergs/s/cm^2/Hz/ster
    I_nu = T_b * 2 * nu ** 2 * kB / cc ** 2

    # convert to Jy/ster
    Jy_ster = I_nu * 1e23

    # convert to Jy/arcsec^2
    Jy_arcsec2 = Jy_ster * arcsec ** 2

    return Jy_arcsec2
