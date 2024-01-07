r"""The ``images`` module provides the core functionality of MPoL via 
:class:`mpol.images.ImageCube`."""

from __future__ import annotations

import numpy as np
import torch
import torch.fft  # to avoid conflicts with old torch.fft *function*
from torch import nn

from typing import Any, Callable

from mpol import utils
from mpol.coordinates import GridCoords


class BaseCube(nn.Module):
    r"""
    A base cube of the same dimensions as the image cube. Designed to use a pixel
    mapping function :math:`f_\mathrm{map}` from the base cube values to the ImageCube
    domain.

    .. math::

        I = f_\mathrm{map}(b)

    The ``base_cube`` pixel values are set as PyTorch `parameters
    <https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html>`_.

    Parameters
    ----------
    coords : :class:`mpol.coordinates.GridCoords`
        an object instantiated from the GridCoords class, containing information about
        the image `cell_size` and `npix`.
    nchan : int
        the number of channels in the base cube. Default = 1.
    pixel_mapping : function
        a PyTorch function mapping the base pixel
        representation to the cube representation. If `None`, defaults to
        `torch.nn.Softplus()`. Output of the function should be in units of
        [:math:`\mathrm{Jy}\,\mathrm{arcsec}^{-2}`].
    base_cube : torch.double tensor, optional
        a pre-packed base cube to initialize
        the model with. If None, assumes ``torch.zeros``. See
        :ref:`cube-orientation-label` for more information on the expectations of
        the orientation of the input image.
    """

    def __init__(
        self,
        coords: GridCoords,
        nchan: int = 1,
        pixel_mapping: Callable[[torch.Tensor], torch.Tensor] | None = None,
        base_cube: torch.Tensor | None = None,
    ) -> None:
        super().__init__()

        self.coords = coords
        self.nchan = nchan

        # The ``base_cube`` is already packed to make the Fourier transformation easier
        if base_cube is None:
            # base_cube = -3 yields a nearly-blank cube after softplus, whereas
            # base_cube = 0.0 yields a cube with avg value of ~0.7, which is too high
            self.base_cube = nn.Parameter(
                -3
                * torch.ones(
                    (self.nchan, self.coords.npix, self.coords.npix),
                    requires_grad=True,
                    dtype=torch.double,
                )
            )

        else:
            # We expect the user to supply a pre-packed base cube
            # so that it's ready to go for the FFT
            # We could apply this transformation for the user, but I think it will
            # lead to less confusion if we make this transformation explicit
            # for the user during the setup phase.
            self.base_cube = nn.Parameter(base_cube, requires_grad=True)

        if pixel_mapping is None:
            self.pixel_mapping: Callable[
                [torch.Tensor], torch.Tensor
            ] = torch.nn.Softplus()
        else:
            self.pixel_mapping = pixel_mapping

    def forward(self) -> torch.Tensor:
        r"""
        Calculate the image representation from the ``base_cube`` using the pixel
        mapping

        .. math::

            I = f_\mathrm{map}(b)

        Returns : an image cube in units of [:math:`\mathrm{Jy}\,\mathrm{arcsec}^{-2}`].
        """

        return self.pixel_mapping(self.base_cube)


class HannConvCube(nn.Module):
    r"""
    This convolutional layer convolves an input cube by a small 3x3 filter with shape

    .. math::

        \begin{bmatrix}
        0.0625 & 0.1250 & 0.0625 \\
        0.1250 & 0.2500 & 0.1250 \\
        0.0625 & 0.1250 & 0.0625 \\
        \end{bmatrix}

    which is the 2D version of the discretely-sampled response function corresponding to
    a Hann window, i.e., it is two 1D Hann windows multiplied together. This is a 
    convolutional kernel in the image plane, and so effectively acts as apodization 
    by a Hann window function in the Fourier domain. For more information, see the 
    following Wikipedia articles on `Window Functions 
    <https://en.wikipedia.org/wiki/Window_function>`_ in general and the `Hann Window 
    <https://en.wikipedia.org/wiki/Hann_function>`_ specifically.

    The idea is that this layer would help naturally attenuate high spatial frequency 
    artifacts by baking in a natural apodization in the Fourier plane.

    Args:
        nchan (int): number of channels
        requires_grad (bool): keep kernel fixed
    """

    def __init__(self, nchan: int, requires_grad: bool = False) -> None:
        super().__init__()
        # simple convolutional filter operates on per-channel basis
        # 3x3 Hann filter
        self.m = nn.Conv2d(
            in_channels=nchan,
            out_channels=nchan,
            kernel_size=3,
            stride=1,
            groups=nchan,
            padding=1,  # required to get same sized output for 3x3 kernel
        )

        # weights has size (nchan, 1, 3, 3)
        # bias has shape (nchan)

        # build out the discretely-sampled Hann kernel
        spec = torch.tensor([0.25, 0.5, 0.25], dtype=torch.double)
        nugget = torch.outer(spec, spec)  # shape (3,3) 2D Hann kernel
        exp = torch.unsqueeze(torch.unsqueeze(nugget, 0), 0)  # shape (1, 1, 3, 3)
        weight = exp.repeat(nchan, 1, 1, 1)  # shape (nchan, 1, 3, 3)

        # set the weight and bias
        self.m.weight = nn.Parameter(
            weight, requires_grad=requires_grad
        )  # set the (untunable) weight

        # set the bias to zero
        self.m.bias = nn.Parameter(
            torch.zeros(nchan, dtype=torch.double), requires_grad=requires_grad
        )

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        r"""Args:
            cube (torch.double tensor, of shape ``(nchan, npix, npix)``): a prepacked
            image cube, for example, from ImageCube.forward()

        Returns:
            torch.complex tensor: the FFT of the image cube, in packed format and of
            shape ``(nchan, npix, npix)``
        """
        # Conv2d is designed to work on batches, so some extra unsqueeze/squeezing
        # action is required.
        # Additionally, the convolution must be done on the *sky-oriented* cube
        sky_cube = utils.packed_cube_to_sky_cube(cube)

        # augment extra "batch" dimension to cube, to make it (1, nchan, npix, npix)
        aug_sky_cube = torch.unsqueeze(sky_cube, dim=0)

        # do convolution
        conv_aug_sky_cube = self.m(aug_sky_cube)

        # remove extra "batch" dimension
        # we're not using unsqueeze here, since there's a possibility that nchan=1
        # and we want to keep that dimension
        conv_sky_cube = conv_aug_sky_cube[0]

        # return in packed format
        return utils.sky_cube_to_packed_cube(conv_sky_cube)


class ImageCube(nn.Module):
    r"""
    The parameter set is the pixel values of the image cube itself. The pixels are
    assumed to represent samples of the specific intensity and are given in units of
    [:math:`\mathrm{Jy}\,\mathrm{arcsec}^{-2}`].

    All keyword arguments are required unless noted. The passthrough argument is
    essential for specifying whether the ImageCube object is the set of root parameters
    (``passthrough==False``) or if its simply a passthrough layer
    (``pasthrough==True``). In either case, ImageCube is essentially an identity layer,
    since no transformations are applied to the ``cube`` tensor. The main purpose of
    the ImageCube layer is to provide useful functionality around the ``cube`` tensor,
    such as returning a sky_cube representation and providing FITS writing
    functionality. In the case of ``passthrough==False``, the ImageCube layer also acts
    as a container for the trainable parameters.

    Parameters
    ----------
    coords : :class:`mpol.coordinates.GridCoords`
        an object instantiated from the GridCoords class, containing information about
        the image `cell_size` and `npix`.
    nchan : int
        the number of channels in the base cube. Default = 1.
    """

    def __init__(
        self,
        coords: GridCoords,
        nchan: int = 1,
    ) -> None:
        super().__init__()

        self.coords = coords
        self.nchan = nchan
        self.register_buffer("packed_cube", None)

    def forward(self, packed_cube: torch.Tensor) -> torch.Tensor:
        r"""
        Pass the cube through as an identity operation, storing the value to the
        internal buffer. After the cube has been passed through, convenience
        instance attributes like `sky_cube` and `flux` will reflect the updated cube.

        Parameters
        ----------
        packed_cube : :class:`torch.Tensor` of type :class:`torch.double`
            3D torch tensor of shape ``(nchan, npix, npix)``) in 'packed' format

        Returns
        -------
        :class:`torch.Tensor` of :class:`torch.double` type
            tensor of shape ``(nchan, npix, npix)``), same as `cube`
        """
        self.packed_cube = packed_cube

        return self.packed_cube

    @property
    def sky_cube(self) -> torch.Tensor:
        """
        The image cube arranged as it would appear on the sky.

        Returns:
            torch.double : 3D image cube of shape ``(nchan, npix, npix)``

        """
        return utils.packed_cube_to_sky_cube(self.packed_cube)

    @property
    def flux(self) -> torch.Tensor:
        """
        The spatially-integrated flux of the image. Returns a 'spectrum' with the flux
        in each channel in units of Jy.

        Returns:
            torch.double: a 1D tensor of length ``(nchan)``
        """

        # convert from Jy/arcsec^2 to Jy/pixel using area of a pixel
        # multiply by arcsec^2/pixel
        return self.coords.cell_size**2 * torch.sum(self.packed_cube, dim=(1, 2))

    def to_FITS(
        self,
        fname: str = "cube.fits",
        overwrite: bool = False,
        header_kwargs: dict | None = None,
    ) -> None:
        """
        Export the image cube to a FITS file.

        Args:
            fname (str): the name of the FITS file to export to.
            overwrite (bool): if the file already exists, overwrite?
            header_kwargs (dict): Extra keyword arguments to write to the FITS header.

        Returns:
            None
        """

        from astropy import wcs
        from astropy.io import fits

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

        hdu = fits.PrimaryHDU(utils.torch2npy(self.sky_cube), header=header)

        hdul = fits.HDUList([hdu])
        hdul.writeto(fname, overwrite=overwrite)

        hdul.close()

    def convolve_packed_cube(
        packed_cube: torch.Tensor,
        coords: GridCoords,
        FWHM_maj: float,
        FWHM_min: float,
        Omega: float,
    ) -> torch.Tensor:
        r"""
        Convolve an image cube with a 2D Gaussian PSF. Operation is carried out in the Fourier domain using a Gaussian taper.

        Parameters
        ----------
        packed_cube : :class:`torch.Tensor` of :class:`torch.double` type
            shape ``(nchan, npix, npix)`` image cube in packed format.
        coords: :class:`mpol.coordinates.GridCoords`
            object indicating image and Fourier grid specifications.
        FWHM_maj: float, units of arcsec
            the FWHH of the Gaussian along the major axis
        FWHM_min: float, units of arcsec
            the FWHM of the Gaussian along the minor axis
        Omega: float, degrees
            the rotation of the major axis of the PSF, in degrees East of North. 0 degrees rotation has the major axis aligned in the East-West direction.
        """
        nchan, npix_m, npix_l = packed_cube.size()
        assert (npix_m == coords.npix) and (
            npix_l == coords.npix
        ), "packed_cube {:} does not have the same pixel dimensions as indicated by coords {:}".format(
            packed_cube.size(), coords.npix
        )

        # in FFT packed format
        # we're round-tripping, so we can ignore prefactors for correctness
        # calling this `vis_like`, since it's not actually the vis
        vis_like = torch.fft.fftn(packed_cube, dim=(1, 2))
        
        # convert FWHM to sigma 
        FWHM2sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
        sigma_x = FWHM_maj * FWHM2sigma
        sigma_y = FWHM_min * FWHM2sigma

        # calculate corresponding uu and vv matrices in packed format
        taper_2D = utils.fourier_gaussian_lambda_arcsec(coords.packed_u_centers_2D, coords.packed_v_centers_2D, a=1.0, delta_x=0.0, delta_y=0.0, sigma_x=sigma_x, sigma_y=sigma_y, Omega=Omega)
        
        # calculate taper on packed image
        tapered_vis = vis_like * torch.broadcast_to(taper_2D, packed_cube.size()) 

        # iFFT back, ignoring prefactors for round-trip
        convolved_packed_cube = torch.fft.fftn(tapered_vis, dim=(1,2)) 

        return convolved_packed_cube
