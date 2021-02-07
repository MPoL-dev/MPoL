import numpy as np
import torch
from torch import nn

from .constants import arcsec
from . import utils


class DatasetConnector(nn.Module):
    r"""
    Connect a FourierCube to the data, and return visibilities for calculating the loss.
    """

    def __init__(self, FourierCube, GriddedDataset, **kwargs):
        super().__init__()

        # check to make sure that the FourierCube and GriddedDataset
        # were both initialized with the same GridCoords settings.
        assert FourierCube.coords == GriddedDataset.coords

    def forward(self, vis, dataset):
        r"""
        Return model and data samples for evaluation with a likelihood function.

        Args: 
            vis: torch tensor 
            dataset: gridded PyTorch dataset w/ mask locations  
        """

        # torch delivers the real and imag components separately
        vis_re = vis[:, :, :, 0]
        vis_im = vis[:, :, :, 1]

        # grid mask is a (nchan, npix, npix//2 + 1) size boolean array
        re = vis_re.masked_select(dataset.grid_mask)
        im = vis_im.masked_select(dataset.grid_mask)

        return re, im
