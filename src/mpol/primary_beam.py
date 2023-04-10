r"""The ``primary_beam`` module provides the core functionality of MPoL via :class:`mpol.fourier.PrimaryBeamCube`."""

from __future__ import annotations

import numpy as np
import torch
import torch.fft  # to avoid conflicts with old torch.fft *function*
import torchkbnufft
from torch import nn

from . import utils
from .coordinates import GridCoords

from .gridding import _check_freq_1d

class PrimaryBeamCube(nn.Module):
    r"""
    A ImageCube representing the primary beam of a described dish type. Currently can correct for a
    uniform or center-obscured dish. The forward() method multiplies an image cube by this primary beam mask.
    
     Args:
        cell_size (float): the width of a pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        nchan (int): the number of channels in the image
        dish_type (string): the type of dish to correct for. Either 'uniform' or 'obscured'.
        dish_radius (float): the radius of the dish (in meters)
        dish_kwargs (dict): any additional arguments needed for special dish types. Currently only uses:
            dish_obscured_radius (float): the radius of the obscured portion of the dish
    """
    def __init__(
        self,
        coords,
        nchan=1,
        chan_freqs=None,
        dish_type=None,
        dish_radius=None,
        **dish_kwargs,
    ):
        super().__init__()
        
        #_setup_coords(self, cell_size, npix, coords, nchan) TODO: update this
        
        _check_freq_1d(chan_freqs)
        assert (chan_freqs is None) or (len(chan_freqs) == nchan), "Length of chan_freqs must be equal to nchan"
        
        assert (dish_type is None) or (dish_type in ["uniform", "obscured"]), "Provided dish_type must be 'uniform' or 'obscured'"
        
        self.coords = coords
        self.nchan = nchan
        
        self.default_mask = nn.Parameter(
            torch.full(
                (self.nchan, self.coords.npix, self.coords.npix),
                fill_value=1.0,
                requires_grad=False,
                dtype=torch.double,
            )
        )
        
        if dish_type is None:
            self.pb_mask = self.default_mask
        elif dish_type == "uniform":
            self.pb_mask = self.uniform_mask(chan_freqs, dish_radius)
        elif dish_type == "obscured":
            self.pb_mask = self.obscured_mask(chan_freqs, dish_radius, **dish_kwargs)

    @classmethod
    def from_image_properties(
        cls, cell_size, npix, nchan=1,
        chan_freqs=None, dish_type=None,
        dish_radius=None, **dish_kwargs
    ) -> ImageCube:
        coords = GridCoords(cell_size, npix)
        return cls(coords, nchan, chan_freqs, dish_type, dish_radius, **dish_kwargs)
    
    def forward(self, cube):
        r"""Args:
            cube (torch.double tensor, of shape ``(nchan, npix, npix)``): a prepacked image cube, for example, from ImageCube.forward()

        Returns:
            (torch.complex tensor, of shape ``(nchan, npix, npix)``): the FFT of the image cube, in packed format.
        """
        return torch.mul(self.pb_mask, cube)
    
    
    def uniform_mask(self, chan_freqs, dish_radius):
        r"""
        Generates airy disk primary beam correction mask.
        """
        assert dish_radius > 0., "Dish radius must be positive"
        ratio = 2. * dish_radius * np.array([[chan_freqs]]).T / 2.998e8

        ratio_cube = np.tile(ratio,(1,self.coords.npix,self.coords.npix))
        r_2D = np.sqrt(self.coords.packed_x_centers_2D**2 + self.coords.packed_y_centers_2D**2)  # arcsec
        r_2D_rads = r_2D * np.pi / 180. / 60. / 60. # radians
        r_cube = np.tile(r_2D_rads,(self.nchan,1,1))

        r_normed_cube = np.pi * r_cube * ratio_cube

        mask = np.where(r_normed_cube > 0.,
                        (2. * j1(r_normed_cube) / r_normed_cube)**2,
                        1.)
        return torch.tensor(mask)
        
    
    def obscured_mask(self, chan_freqs, dish_radius, dish_obscured_radius=None, **extra_kwargs):
        r"""
        Generates airy disk primary beam correction mask.
        """
        assert dish_obscured_radius is not None, "Obscured dish requires kwarg 'dish_obscured_radius'"
        assert dish_radius > 0., "Dish radius must be positive"
        assert dish_obscured_radius > 0., "Obscured dish radius must be positive"
        assert dish_radius > dish_obscured_radius, "Primary dish radius must be greater than obscured radius"
        
        ratio = 2. * dish_radius * np.array([[chan_freqs]]).T / 2.998e8
        ratio_cube = np.tile(ratio,(1,self.coords.npix,self.coords.npix))
        r_2D = np.sqrt(self.coords.packed_x_centers_2D**2 + self.coords.packed_y_centers_2D**2)  # arcsec
        r_2D_rads = r_2D * np.pi / 180. / 60. / 60. # radians
        r_cube = np.tile(r_2D_rads,(self.nchan,1,1))
        
        eps = dish_obscured_radius / dish_radius
        r_normed_cube = np.pi * r_cube * ratio_cube
        
        norm_factor = (1.-eps**2)**2
        mask = np.where(r_normed_cube > 0.,
                        (j1(r_normed_cube) / r_normed_cube 
                                    - eps*j1(eps*r_normed_cube) / r_normed_cube)**2 / norm_factor,
                        1.)
        return torch.tensor(mask)
        
    @property
    def sky_cube(self):
        """
        The primary beam mask arranged as it would appear on the sky.

        Returns:
            torch.double : 3D image cube of shape ``(nchan, npix, npix)``

        """
        return utils.packed_cube_to_sky_cube(self.pb_mask)

    def to_FITS(self, fname="cube.fits", overwrite=False, header_kwargs=None):
        """
        Export the primary beam cube to a FITS file.

        Args:
            fname (str): the name of the FITS file to export to.
            overwrite (bool): if the file already exists, overwrite?
            header_kwargs (dict): Extra keyword arguments to write to the FITS header.

        Returns:
            None
        """

        try:
            from astropy import wcs
            from astropy.io import fits
        except ImportError:
            print(
                "Please install the astropy package to use FITS export functionality."
            )

        w = wcs.WCS(naxis=2)

        w.wcs.crpix = np.array([1, 1])
        w.wcs.cdelt = (
            np.array([self.coords.cell_size, self.coords.cell_size]) / 3600
        )  # decimal degrees
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        header = w.to_header()

        # add in the kwargs to the header
        if header_kwargs is not None:
            for k, v in header_kwargs.items():
                header[k] = v

        hdu = fits.PrimaryHDU(self.pb_mask.detach().cpu().numpy(), header=header)

        hdul = fits.HDUList([hdu])
        hdul.writeto(fname, overwrite=overwrite)

        hdul.close()