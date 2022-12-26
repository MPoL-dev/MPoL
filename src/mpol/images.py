r"""The ``images`` module provides the core functionality of MPoL via :class:`mpol.images.ImageCube`."""

import numpy as np
import torch
import torch.fft  # to avoid conflicts with old torch.fft *function*
from torch import nn

from . import utils
from .coordinates import GridCoords
from .gridding import _setup_coords


class BaseCube(nn.Module):
    r"""
    A base cube of the same dimensions as the image cube. Designed to use a pixel mapping function :math:`f_\mathrm{map}` from the base cube values to the ImageCube domain.

    .. math::

        I = f_\mathrm{map}(b)

    The ``base_cube`` pixel values are set as PyTorch `parameters <https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html>`_.

    Args:
        cell_size (float): the width of a pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        nchan (int): the number of channels in the base cube. Default = 1.
        pixel_mapping (torch.nn): a PyTorch function mapping the base pixel representation to the cube representation. If `None`, defaults to `torch.nn.Softplus() <https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html#torch.nn.Softplus>`_. Output of the function should be in units of [:math:`\mathrm{Jy}\,\mathrm{arcsec}^{-2}`].
        base_cube (torch.double tensor, optional): a pre-packed base cube to initialize the model with. If None, assumes ``torch.zeros``. See :ref:`cube-orientation-label` for more information on the expectations of the orientation of the input image.
    """

    def __init__(
        self,
        cell_size=None,
        npix=None,
        coords=None,
        nchan=None,
        pixel_mapping=None,
        base_cube=None,
    ):

        super().__init__()
        _setup_coords(self, cell_size, npix, coords, nchan)

        # The ``base_cube`` is already packed to make the Fourier transformation easier
        if base_cube is None:
            self.base_cube = nn.Parameter(
                torch.full(
                    (self.nchan, self.coords.npix, self.coords.npix),
                    fill_value=0.05,
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
            self.pixel_mapping = torch.nn.Softplus()
        else:
            # TODO assert that this is a PyTorch function
            self.pixel_mapping = pixel_mapping

    def forward(self):
        r"""
        Calculate the image representation from the ``base_cube`` using the pixel mapping

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

    which is the 2D version of the discretely-sampled response function corresponding to a Hann window, i.e., it is two 1D Hann windows multiplied together. This is a convolutional kernel in the image plane, and so effectively acts as apodization by a Hann window function in the Fourier domain. For more information, see the following Wikipedia articles on `Window Functions <https://en.wikipedia.org/wiki/Window_function>`_ in general and the `Hann Window <https://en.wikipedia.org/wiki/Hann_function>`_ specifically.

    The idea is that this layer would help naturally attenuate high spatial frequency artifacts by baking in a natural apodization in the Fourier plane.

    Args:
        nchan (int): number of channels
        requires_grad (bool): keep kernel fixed
    """

    def __init__(self, nchan, requires_grad=False):

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

    def forward(self, cube):
        r"""Args:
            cube (torch.double tensor, of shape ``(nchan, npix, npix)``): a prepacked image cube, for example, from ImageCube.forward()

        Returns:
            torch.complex tensor: the FFT of the image cube, in packed format and of shape ``(nchan, npix, npix)``
        """
        # Conv2d is designed to work on batches, so some extra unsqueeze/squeezing action is required.
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
    The parameter set is the pixel values of the image cube itself. The pixels are assumed to represent samples of the specific intensity and are given in units of [:math:`\mathrm{Jy}\,\mathrm{arcsec}^{-2}`].

    All keyword arguments are required unless noted. The passthrough argument is essential for specifying whether the ImageCube object is the set of root parameters (``passthrough==False``) or if its simply a passthrough layer (``pasthrough==True``). In either case, ImageCube is essentially an identity layer, since no transformations are applied to the ``cube`` tensor. The main purpose of the ImageCube layer is to provide useful functionality around the ``cube`` tensor, such as returning a sky_cube representation and providing FITS writing functionility. In the case of ``passthrough==False``, the ImageCube layer also acts as a container for the trainable parameters.

    Args:
        cell_size (float): the width of a pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        nchan (int): the number of channels in the image
        passthrough (bool): if passthrough, assume ImageCube is just a layer as opposed to parameter base.
        cube (torch.double tensor, of shape ``(nchan, npix, npix)``): (optional) a prepacked image cube to initialize the model with in units of [:math:`\mathrm{Jy}\,\mathrm{arcsec}^{-2}`]. If None, assumes starting ``cube`` is ``torch.zeros``. See :ref:`cube-orientation-label` for more information on the expectations of the orientation of the input image.
    """

    def __init__(
        self,
        cell_size=None,
        npix=None,
        coords=None,
        nchan=None,
        passthrough=False,
        cube=None,
    ):
        super().__init__()
        _setup_coords(self, cell_size, npix, coords, nchan)

        self.passthrough = passthrough

        if not self.passthrough:
            if cube is None:
                self.cube = nn.Parameter(
                    torch.full(
                        (self.nchan, self.coords.npix, self.coords.npix),
                        fill_value=0.0,
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
                self.cube = nn.Parameter(cube)
        else:
            # ImageCube is working as a passthrough layer, so cube should
            # only be provided as an arg to the forward method, not as
            # an initialization argument
            self.cube = None

    def forward(self, cube=None):
        r"""
        If the ImageCube object was initialized with ``passthrough=True``, the ``cube`` argument is required. ``forward`` essentially just passes this on as an identity operation.

        If the ImageCube object was initialized with ``passthrough=False``, the ``cube`` argument is not permitted, and ``forward`` passes on the stored ``nn.Parameter`` cube as an identity operation.

        Args:
            cube (3D torch tensor of shape ``(nchan, npix, npix)``): only permitted if the ImageCube object was initialized with ``passthrough=True``.

        Returns: (3D torch.double tensor of shape ``(nchan, npix, npix)``) as identity operation
        """

        if cube is not None:
            assert (
                self.passthrough
            ), "ImageCube.passthrough must be True if supplying cube."
            self.cube = cube

        if not self.passthrough:
            assert cube is None, "Do not supply cube if ImageCube.passthrough == False."

        return self.cube

    @property
    def sky_cube(self):
        """
        The image cube arranged as it would appear on the sky.

        Returns:
            torch.double : 3D image cube of shape ``(nchan, npix, npix)``

        """
        return utils.packed_cube_to_sky_cube(self.cube)

    def to_FITS(self, fname="cube.fits", overwrite=False, header_kwargs=None):
        """
        Export the image cube to a FITS file.

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

        hdu = fits.PrimaryHDU(self.sky_cube.detach().cpu().numpy(), header=header)

        hdul = fits.HDUList([hdu])
        hdul.writeto(fname, overwrite=overwrite)

        hdul.close()
