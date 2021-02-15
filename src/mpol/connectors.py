import numpy as np
import torch
from torch import nn

from .constants import arcsec
from . import utils


class DatasetConnector(nn.Module):
    r"""
    Connect a FourierCube to the data, and return indexed model visibilities for calculating the loss. 

    Args:
        FourierCube: instantiated :class:`~mpol.images.FourierCube` object
        GriddedDataset: instantiated :class:`~mpol.datasets.GriddedDataset` object
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
        Return model samples for evaluation with a likelihood function.

        Args: 
            vis (torch complex tensor): torch tensor ``(nchan, npix, npix)`` shape to be indexed by the ``mask`` from :class:`~mpol.datasets.GriddedDataset`.

        Returns (torch complex tensor):  1d torch tensor of model samples collapsed across cube dimensions like ``vis_indexed`` and ``weight_indexed`` of :class:`~mpol.datasets.GriddedDataset`
        """

        assert (
            vis.size()[0] == self.mask.size()[0]
        ), "vis and dataset mask do not have the same number of channels."

        # As of Pytorch 1.7.0, complex numbers are partially supported.
        # However, masked_select does not yet work (with gradients)
        # on the complex vis, so hence this awkward step of selecting
        # the reals and imaginaries separately
        re = vis.real.masked_select(self.mask)
        im = vis.imag.masked_select(self.mask)

        # we had trouble returning things as re + 1.0j * im,
        # but for some reason torch.complex seems to work OK.
        return torch.complex(re, im)

