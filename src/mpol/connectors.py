import numpy as np
import torch
from torch import nn

from .constants import arcsec
from . import images
from . import utils


class GriddedDatasetConnector(nn.Module):
    r"""
    Connect a FourierCube to the gridded dataset, and return indexed model visibilities for calculating the loss. 

    Args:
        fourierCube: instantiated :class:`~mpol.images.FourierCube` object
        griddedDataset: instantiated :class:`~mpol.datasets.GriddedDataset` object
    """

    def __init__(self, fourierCube, griddedDataset):
        super().__init__()

        # check to make sure that the FourierCube and GriddedDataset
        # were both initialized with the same GridCoords settings.
        assert fourierCube.coords == griddedDataset.coords

        self.fourierCube = fourierCube
        self.griddedDataset = griddedDataset
        self.coords = fourierCube.coords

        # take the mask
        self.mask = griddedDataset.mask

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


class GriddedResidualConnector(GriddedDatasetConnector):
    r"""
    Calculate residual gridded products.
    """

    def forward(self):
        r"""Calculate the residuals as 
        
        ..math::
        
            \mathrm{data} - \mathrm{model}

        And produce a dictionary of image cube products. Cubes is pre-packed.

        Returns: None

        Residual products are available as PyTorch tensor instance attributes after forward call.

        :ivar cube: pre-packed image cube
        :ivar residuals: pre-packed (complex) residuals
        :ivar amp: pre-packed amplitude
        :ivar phase: pre-packed phase
        
        Real and imaginary components of the residuals can be accessed directly via ``residuals.real`` and ``residuals.imag``.

        """
        self.residuals = self.griddedDataset.vis_gridded - self.fourierCube.vis

        self.amp = torch.abs(self.residuals)
        self.phase = torch.angle(self.residuals)

        # calculate the correpsonding residual dirty image (under uniform weighting)
        cube = self.coords.npix ** 2 * torch.fft.ifftn(self.residuals, dim=(1, 2))

        assert (
            torch.max(cube.imag) < 1e-10
        ), "Dirty image contained substantial imaginary values, check input visibilities, otherwise raise a github issue."

        self.cube = cube.real

    @property
    def sky_cube(self):
        """
        The image cube arranged as it would appear on the sky.

        Returns:
            torch.double : 3D image cube of shape ``(nchan, npix, npix)``
            
        """
        return images.packed_cube_to_sky_cube(self.cube)
