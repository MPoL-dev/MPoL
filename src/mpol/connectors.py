import numpy as np
import torch
from torch import nn

from .constants import arcsec
from . import utils


class DatasetConnector(nn.Module):
    r"""
    Connect an image cube to the data.
    """

    def __init__(self, cell_size=None, npix=None, coords=None, **kwargs):
        # Set up with the gridcoordinates.

        super().__init__()
        pass

    def forward(self, cube, dataset):
        r"""
        Return model and data samples for evaluation with a likelihood function.

        Args: 
            cube: torch tensor to convert to dataset
            dataset: gridded PyTorch dataset w/ locations
        """

        # convert the image to Jy/ster and perform the RFFT
        # the self.cell_size prefactor (in radians) is to obtain the correct output units
        # since it needs to correct for the spacing of the input grid.
        # See MPoL documentation and/or TMS Eqn A8.18 for more information.
        # Alternatively we could send the rfft routine _cube in its native Jy/arcsec^2
        # and then multiply the result by cell_size (in units of arcsec).
        # It seemed easiest to do it this way where we keep things in radians.
        self._vis = self.cell_size ** 2 * torch.rfft(
            self._cube / arcsec ** 2, signal_ndim=2
        )

        # torch delivers the real and imag components separately
        vis_re = self._vis[:, :, :, 0]
        vis_im = self._vis[:, :, :, 1]

        # grid mask is a (nchan, npix, npix//2 + 1) size boolean array
        re = vis_re.masked_select(dataset.grid_mask)
        im = vis_im.masked_select(dataset.grid_mask)

    @property
    def vis(self):
        r"""
        The visibility RFFT cube fftshifted for plotting with ``imshow`` (the v coordinate goes from -ve to +ve).

        Returns:
            torch.double: visibility cube
        """

        return utils.fftshift(self._vis, axes=(1,))

    @property
    def psd(self):
        r"""
        The power spectral density of the cube, fftshifted for plotting. (The v coordinate goes from -ve to +ve).

        Returns:
            torch.double: power spectral density cube
        """

        vis_re = self.vis[:, :, :, 0]
        vis_im = self.vis[:, :, :, 1]

        psd = vis_re ** 2 + vis_im ** 2

        return psd
