import numpy as np
import torch
from torch import nn
import torch.fft  # to avoid conflicts with old torch.fft *function*

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
            vis (torch complex tensor): torch tensor ``(nchan, npix, npix)`` shape to be indexed by the ``mask`` from :class:`~mpol.datasets.GriddedDataset`. Assumes tensor is "pre-packed."

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
    Connect a FourierCube to the gridded dataset and calculate residual products useful for visualization and debugging. The products are available as instance and property attributes after the ``forward`` call.

    Args:
        fourierCube: instantiated :class:`~mpol.images.FourierCube` object
        griddedDataset: instantiated :class:`~mpol.datasets.GriddedDataset` object
    """

    def forward(self):
        r"""Calculate the residuals as 
        
        .. math::
        
            \mathrm{residuals} = \mathrm{data} - \mathrm{model}

        And store residual products as PyTorch tensor instance and property attributes. 
        """
        self.residuals = self.griddedDataset.vis_gridded - self.fourierCube.vis

        self.amp = torch.abs(self.residuals)
        self.phase = torch.angle(self.residuals)

        # calculate the correpsonding residual dirty image (under uniform weighting).
        # see units_and_conventions.rst for the calculation of the prefactors.
        # But essentially this calculation requires a prefactor of npix**2 * uv_cell_size**2
        # Assuming uv_cell_size would be measured in units of cycles/arcsec, then this prefactor is
        # equivalent to 1/cell_size**2, where cell_size is units of arcsec
        cube = (
            1
            / (self.coords.cell_size ** 2)
            * torch.fft.ifftn(self.residuals, dim=(1, 2))
        )  # Jy/arcsec^2

        assert (
            torch.max(cube.imag) < 1e-10
        ), "Dirty image contained substantial imaginary values, check input visibilities, otherwise raise a github issue."

        self.cube = cube.real

    @property
    def sky_cube(self):
        """
        The image cube arranged as it would appear on the sky. Array dimensions for plotting given by ``self.coords.img_ext``.

        Returns:
            torch.double : 3D image cube of shape ``(nchan, npix, npix)`` in units of [:math:`\mathrm{Jy}\,\mathrm{arcsec}^{-2}`].
            
        """
        return images.packed_cube_to_sky_cube(self.cube)

    @property
    def sky_amp(self):
        """
        The amplitude of the residuals, arranged in unpacked format corresponding to the FFT of the sky_cube. Array dimensions for plotting given by ``self.coords.vis_ext``.

        Returns:
            torch.double : 3D amplitude cube of shape ``(nchan, npix, npix)``
        """
        return images.packed_cube_to_sky_cube(self.amp)

    @property
    def sky_phase(self):
        """
        The phase of the residuals, arranged in unpacked format corresponding to the FFT of the sky_cube. Array dimensions for plotting given by ``self.coords.vis_ext``.

        Returns:
            torch.double : 3D phase cube of shape ``(nchan, npix, npix)``
        """
        return images.packed_cube_to_sky_cube(self.phase)

