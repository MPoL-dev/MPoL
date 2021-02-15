import numpy as np
import torch
from torch import nn

from .constants import arcsec
from . import utils


class DatasetConnector(nn.Module):
    r"""
    Connect a FourierCube to the data, and return visibilities for calculating the loss. 

    Args:
        FourierCube: 
        GriddedDataset:
    """

    def __init__(self, FourierCube, GriddedDataset):
        super().__init__()

        # check to make sure that the FourierCube and GriddedDataset
        # were both initialized with the same GridCoords settings.
        assert FourierCube.coords == GriddedDataset.coords

        # take the mask
        self.mask = GriddedDataset.mask

    def forward(self, vis):
        r"""
        Return model and data samples for evaluation with a likelihood function.

        Args: 
            vis: torch tensor old representation 
            dataset: gridded PyTorch dataset w/ mask locations  
        """

        # grid_mask is a (nchan, npix, npix) boolean array

        # # torch delivers the real and imag components separately
        # vis_re = vis[:, :, :, 0]
        # vis_im = vis[:, :, :, 1]

        re = vis.real.masked_select(self.mask)
        im = vis.imag.masked_select(self.mask)
        return re + 1.0j * im

