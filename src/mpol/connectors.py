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

    def __init__(self, FourierCube, GriddedDataset, **kwargs):
        super().__init__()

        # check to make sure that the FourierCube and GriddedDataset
        # were both initialized with the same GridCoords settings.
        assert FourierCube.coords == GriddedDataset.coords

    def forward(self, vis, dataset):
        r"""
        Return model and data samples for evaluation with a likelihood function.

        Args: 
            vis: complex torch tensor 
            dataset: gridded PyTorch dataset w/ mask locations  
        """

        # grid_mask is a (nchan, npix, npix) boolean array
        re = vis.real.masked_select(dataset.mask)
        im = vis.imag.masked_select(dataset.mask)
        return re + 1.0j * im

