r"""The ``images`` module provides the core functionality of MPoL via
:class:`mpol.images.ImageCube`."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
import torch.fft  # to avoid conflicts with old torch.fft *function*
from torch import nn
import math

from mpol import constants, utils
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
        spec = torch.tensor([0.25, 0.5, 0.25])
        nugget = torch.outer(spec, spec)  # shape (3,3) 2D Hann kernel
        exp = torch.unsqueeze(torch.unsqueeze(nugget, 0), 0)  # shape (1, 1, 3, 3)
        weight = exp.repeat(nchan, 1, 1, 1)  # shape (nchan, 1, 3, 3)

        # set the weight and bias
        self.m.weight = nn.Parameter(
            weight, requires_grad=requires_grad
        )  # set the (untunable) weight

        # set the bias to zero
        self.m.bias = nn.Parameter(torch.zeros(nchan), requires_grad=requires_grad)

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


class GaussBaseBeam(nn.Module):
    r"""
    This layer will convolve the base cube with a Gaussian beam of variable resolution.
    The FWHM of the beam (in arcsec) is a trainable parameter of the layer.

    Parameters
    ----------
    coords : :class:`mpol.coordinates.GridCoords`
        an object instantiated from the GridCoords class, containing information about
        the image `cell_size` and `npix`.
    nchan : int
        the number of channels in the base cube. Default = 1.
    FWHM: float, units of arcsec
        the FWHH of the Gaussian
    """

    def __init__(self, coords: GridCoords, nchan: int) -> None:
        super().__init__()
        
        self.coords = coords 
        self.nchan = nchan
        
        self._FWHM_base = nn.Parameter(torch.tensor([-3.0]))
        self.softplus = nn.Softplus()
        # -3.0 corresponds to about 0.05 arcsec

        # store coordinates to register so they transfer to GPU
        self.register_buffer("u", torch.tensor(self.coords.packed_u_centers_2D, dtype=torch.float32))
        self.register_buffer("v", torch.tensor(self.coords.packed_v_centers_2D, dtype=torch.float32))

    @property
    def FWHM(self):
        r"""Map from base parameter to actual FWHM."""
        return self.softplus(self._FWHM_base)  # ensures always positive

    def forward(self, packed_cube):
        r"""
        Convolve a packed_cube image with a 2D Gaussian PSF. Operation is carried out 
        in the Fourier domain using a Gaussian taper.

        Parameters
        ----------
        packed_cube : :class:`torch.Tensor`  type
            shape ``(nchan, npix, npix)`` image cube in packed format.

        Returns
        -------
        :class:`torch.Tensor`
            The convolved cube in packed format.
        """
        nchan, npix_m, npix_l = packed_cube.size()
        assert (
            (npix_m == self.coords.npix) and (npix_l == self.coords.npix)
        ), "packed_cube {:} does not have the same pixel dimensions as indicated by coords {:}".format(
            packed_cube.size(), self.coords.npix
        )

        # in FFT packed format
        # we're round-tripping, so we can ignore prefactors for correctness
        # calling this `vis_like`, since it's not actually the vis
        vis_like = torch.fft.fftn(packed_cube, dim=(1, 2))

        # convert FWHM to sigma and to radians
        FWHM2sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
        sigma = self.FWHM * FWHM2sigma * constants.arcsec  # radians
    
        # calculate the UV taper from the FWHM size.
        taper_2D = torch.exp(-2 * np.pi**2 * (sigma**2 * self.u**2 + sigma**2 * self.v**2))

        # apply taper to packed image
        tapered_vis = vis_like * torch.broadcast_to(taper_2D, packed_cube.size())

        # iFFT back, ignoring prefactors for round-trip
        convolved_packed_cube = torch.fft.ifftn(tapered_vis, dim=(1, 2))

        # assert imaginaries are effectively zero, otherwise something went wrong
        thresh = 1e-7
        assert (
            torch.max(convolved_packed_cube.imag) < thresh
        ), "Round-tripped image contains max imaginary value {:} > {:} threshold, something may be amiss.".format(
            torch.max(convolved_packed_cube.imag), thresh
        )

        r_cube: torch.Tensor = convolved_packed_cube.real
        return r_cube


class GaussConvCube(nn.Module):
    r"""
    Once instantiated, this convolutional layer is used to convolve the input cube with
    a 2D Gaussian filter. The filter is the same for all channels in the input cube.

    Parameters
    ----------
    coords : :class:`mpol.coordinates.GridCoords`
        an object instantiated from the GridCoords class, containing information about
        the image `cell_size` and `npix`.
    nchan : int
        the number of channels in the base cube. Default = 1.
    FWHM_maj: float, units of arcsec
        the FWHH of the Gaussian along the major axis
    FWHM_min: float, units of arcsec
        the FWHM of the Gaussian along the minor axis
    Omega: float, degrees
        the rotation of the major axis of the PSF, in degrees East of North. 0 degrees rotation has the major axis aligned in the North-South direction.
    requires_grad : bool
        keep kernel fixed
    """

    def __init__(
        self,
        coords: GridCoords,
        nchan: int,
        FWHM_maj: float,
        FWHM_min: float,
        Omega: float = 0,
        requires_grad: bool = False,
    ) -> None:
        super().__init__()

        # convert FWHM to sigma and to radians
        FWHM2sigma = 1 / (2 * np.sqrt(2 * np.log(2)))

        # In this routine, x, y are used in the same sense as the GridCoords
        # object uses 'sky_x' and 'sky_y', i.e. x is l in arcseconds and
        # y is m in arcseconds.

        # assumes major axis along m direction at 0 degrees rotation.
        sigma_y = FWHM_maj * FWHM2sigma  # arcsec
        sigma_x = FWHM_min * FWHM2sigma  # arcsec

        # calculate filter out to some Gaussian width, and make a kernel with an
        # odd number of pixels
        limit = 3.0 * sigma_y
        npix_kernel = 1 + 2 * math.ceil(limit / coords.cell_size)

        if npix_kernel < 3:
            raise RuntimeError(
                """FWHM_maj is so small ({:} arcsec) relative to the 
                cell_size ({:} arcsec) that the convolutional kernel would only be
                               one pixel wide. Increase FWHM_maj or remove this 
                               convolutional layer entirely""".format(
                    npix_kernel, coords.cell_size
                )
            )

        # create a grid to evaluate the 2D Gaussian, using an even number of
        # pixels with the kernel centered (no max pixel)
        kernel_centers = np.linspace(-limit, limit, num=npix_kernel)  # [arcsec]

        # borrowed from GridCoords logic
        x_centers_2D = np.tile(kernel_centers, (npix_kernel, 1))  # [arcsec]
        sky_x_centers_2D = np.fliplr(x_centers_2D)

        sky_y_centers_2D = np.tile(kernel_centers, (npix_kernel, 1)).T  # [arcsec]

        # evaluate Gaussian over grid
        gauss = utils.sky_gaussian_arcsec(
            sky_x_centers_2D,
            sky_y_centers_2D,
            1.0,
            delta_x=0.0,
            delta_y=0.0,
            sigma_x=sigma_x,
            sigma_y=sigma_y,
            Omega=Omega,
        )
        # normalize kernel to keep total flux the same
        gauss /= np.sum(gauss)
        nugget = torch.tensor(gauss, dtype=torch.float32)
        exp = torch.unsqueeze(
            torch.unsqueeze(nugget, 0), 0
        )  # shape (1, 1, npix_kernel, npix_kernel)
        weight = exp.repeat(
            nchan, 1, 1, 1
        )  # shape (nchan, 1, npix_kernel, npix_kernel)

        # groups = nchan will give us the minimal set of filters we need
        # somewhat confusingly, the neural network literature calls this
        # a "depthwise" convolution. I think that "depthwise" is not meant to imply
        # that there is now a consideration of the depth (e.g., color channel)
        # dimension when before there wasn't.
        # Rather, the emphasis is on the *wise*, as in "pairwise," in that
        # each depth channel is treated individually with its own filter, rather than
        # a filter that draws from multiple depth channels at once.
        # I think a better name is "channel-separate" convolution as indicated in the
        # "Understanding Deep Learning" textbook by Prince in Ch 10.6.

        # simple convolutional filter operates on per-channel basis
        self.m = nn.Conv2d(
            in_channels=nchan,
            out_channels=nchan,
            kernel_size=npix_kernel,
            stride=1,
            groups=nchan,
            padding="same",
        )

        # weights has size (nchan, 1, npix_kernel, npix_kernel)
        # bias has shape (nchan)

        # set the weight and bias
        self.m.weight = nn.Parameter(
            weight, requires_grad=requires_grad
        )  # set the (untunable) weight

        # set the bias to zero
        self.m.bias = nn.Parameter(torch.zeros(nchan), requires_grad=requires_grad)

    def forward(self, sky_cube: torch.Tensor) -> torch.Tensor:
        r"""Args:
            sky_cube (torch.double tensor, of shape ``(nchan, npix, npix)``): an image cube in sky format (note, not packed).

        Returns:
            torch.complex tensor: the FFT of the image cube, in sky format and of
            shape ``(nchan, npix, npix)``
        """
        convolved_sky: torch.Tensor
        convolved_sky = self.m(sky_cube)
        return convolved_sky


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
        :class:`torch.Tensor`
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


def uv_gaussian_taper(
    coords: GridCoords, FWHM_maj: float, FWHM_min: float, Omega: float
) -> torch.Tensor:
    r"""
    Compute a packed Gaussian taper in the Fourier domain, to multiply against a packed
    visibility cube. While similar to :meth:`mpol.utils.fourier_gaussian_lambda_arcsec`,
    this routine delivers a visibility-plane taper with maximum amplitude normalized
    to 1.0.

    Parameters
    ----------
    coords: :class:`mpol.coordinates.GridCoords`
        object indicating image and Fourier grid specifications.
    FWHM_maj: float, units of arcsec
        the FWHH of the Gaussian along the major axis
    FWHM_min: float, units of arcsec
        the FWHM of the Gaussian along the minor axis
    Omega: float, degrees
        the rotation of the major axis of the PSF, in degrees East of North. 0 degrees rotation has the major axis aligned in the East-West direction.

    Returns
    -------
    :class:`torch.Tensor` , shape ``(npix, npix)``
        The Gaussian taper in packed format.
    """

    # convert FWHM to sigma and to radians
    FWHM2sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
    sigma_l = FWHM_maj * FWHM2sigma * constants.arcsec  # radians
    sigma_m = FWHM_min * FWHM2sigma * constants.arcsec  # radians

    u = coords.packed_u_centers_2D
    v = coords.packed_v_centers_2D

    # calculate primed rotated coordinates
    Omega_d = Omega * constants.deg
    up = u * np.cos(Omega_d) - v * np.sin(Omega_d)
    vp = u * np.sin(Omega_d) + v * np.cos(Omega_d)

    # calculate the Fourier Gaussian
    taper_2D = np.exp(-2 * np.pi**2 * (sigma_l**2 * up**2 + sigma_m**2 * vp**2))

    # # the fourier_gaussian_lambda_arcsec routine assumes the amplitude
    # # is 1.0 *in the image plane*. This is not the same as having an
    # # amplitude 1.0 in the visibility plane, which is a requirement of a
    # # flux-conserving taper. So we renormalize.
    # taper_2D /= np.max(np.abs(taper_2D))

    return torch.from_numpy(taper_2D)


def convolve_packed_cube(
    packed_cube: torch.Tensor,
    coords: GridCoords,
    FWHM_maj: float,
    FWHM_min: float,
    Omega: float = 0,
) -> torch.Tensor:
    r"""
    Convolve an image cube with a 2D Gaussian PSF. Operation is carried out in the Fourier domain using a Gaussian taper.

    Parameters
    ----------
    packed_cube : :class:`torch.Tensor`  type
        shape ``(nchan, npix, npix)`` image cube in packed format.
    coords: :class:`mpol.coordinates.GridCoords`
        object indicating image and Fourier grid specifications.
    FWHM_maj: float, units of arcsec
        the FWHH of the Gaussian along the major axis
    FWHM_min: float, units of arcsec
        the FWHM of the Gaussian along the minor axis
    Omega: float, degrees
        the rotation of the major axis of the PSF, in degrees East of North. 0 degrees rotation has the major axis aligned in the East-West direction.

    Returns
    -------
    :class:`torch.Tensor`
        The convolved cube in packed format.
    """
    nchan, npix_m, npix_l = packed_cube.size()
    assert (
        (npix_m == coords.npix) and (npix_l == coords.npix)
    ), "packed_cube {:} does not have the same pixel dimensions as indicated by coords {:}".format(
        packed_cube.size(), coords.npix
    )

    # in FFT packed format
    # we're round-tripping, so we can ignore prefactors for correctness
    # calling this `vis_like`, since it's not actually the vis
    vis_like = torch.fft.fftn(packed_cube, dim=(1, 2))

    taper_2D = uv_gaussian_taper(coords, FWHM_maj, FWHM_min, Omega)
    # calculate taper on packed image
    tapered_vis = vis_like * torch.broadcast_to(taper_2D, packed_cube.size())

    # iFFT back, ignoring prefactors for round-trip
    convolved_packed_cube = torch.fft.ifftn(tapered_vis, dim=(1, 2))

    # assert imaginaries are effectively zero, otherwise something went wrong
    thresh = 1e-7
    assert (
        torch.max(convolved_packed_cube.imag) < thresh
    ), "Round-tripped image contains max imaginary value {:} > {:} threshold, something may be amiss.".format(
        torch.max(convolved_packed_cube.imag), thresh
    )

    r_cube: torch.Tensor = convolved_packed_cube.real
    return r_cube
