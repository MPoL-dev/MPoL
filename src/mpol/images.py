r"""The ``images`` module provides the core functionality of MPoL via :class:`mpol.images.ImageCube`."""

import numpy as np
import torch
from torch import nn
import torch.fft  # to avoid conflicts with old torch.fft *function*

from .constants import arcsec
from .gridding import GridCoords, _setup_coords
from . import utils


def sky_cube_to_packed_cube(sky_cube):
    # If it's an identity layer, just set parameters to cube
    flipped = torch.flip(sky_cube, (2,))
    shifted = torch.fft.fftshift(flipped, dim=(1, 2))
    return shifted


def packed_cube_to_sky_cube(packed_cube):
    # fftshift the image cube to the correct quadrants
    shifted = torch.fft.fftshift(packed_cube, dim=(1, 2))
    # flip so that east points left
    flipped = torch.flip(shifted, (2,))
    return flipped


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
        base_cube (torch.double tensor, optional): a pre-packed base cube to initialize the model with. If None, assumes ``torch.zeros``.
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
        cube (torch.double tensor, of shape ``(nchan, npix, npix)``): (optional) a prepacked image cube to initialize the model with in units of [:math:`\mathrm{Jy}\,\mathrm{arcsec}^{-2}`]. If None, assumes starting ``cube`` is ``torch.zeros``. 
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
        return packed_cube_to_sky_cube(self.cube)

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
            from astropy.io import fits
            from astropy import wcs
        except ImportError:
            print(
                "Please install the astropy package to use FITS export functionality."
            )

        w = wcs.WCS(naxis=2)

        w.wcs.crpix = np.array([1, 1])
        w.wcs.cdelt = (
            np.array([self.coords.cell_size, self.coords.cell_size]) * 180.0 / np.pi
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


class FourierCube(nn.Module):
    r"""
    A layer holding the cube corresponding to the FFT of ImageCube.

    Args:
        cell_size (float): the width of an image-plane pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
    """

    def __init__(self, cell_size=None, npix=None, coords=None):

        super().__init__()

        # we don't want to bother with the nchan argument here, so
        # we don't use the convenience method _setup_coords
        # and just do it manually
        if coords:
            assert (
                npix is None and cell_size is None
            ), "npix and cell_size must be empty if precomputed GridCoords are supplied."
            self.coords = coords

        elif npix or cell_size:
            assert (
                coords is None
            ), "GridCoords must be empty if npix and cell_size are supplied."

            self.coords = GridCoords(cell_size=cell_size, npix=npix)

    def forward(self, cube):
        """
        Perform the FFT of the image cube for each channel.
        
        Args:
            cube (torch.double tensor, of shape ``(nchan, npix, npix)``): a prepacked image cube, for example, from ImageCube.forward()

        Returns: 
            (torch.complex tensor, of shape ``(nchan, npix, npix)``): the FFT of the image cube, in packed format. 
        """

        # make sure the cube is 3D
        assert cube.dim() == 3, "cube must be 3D"

        # the self.cell_size prefactor (in arcsec) is to obtain the correct output units
        # since it needs to correct for the spacing of the input grid.
        # See MPoL documentation and/or TMS Eqn A8.18 for more information.
        self.vis = self.coords.cell_size ** 2 * torch.fft.fftn(cube, dim=(1, 2))

        return self.vis

    @property
    def psd(self):
        r"""
        The power spectral density of the cube, in packed format.

        Returns:
            torch.double: power spectral density cube
        """
        return np.abs(self.vis) ** 2

    @property
    def sky_vis(self):
        r"""
        The visibility FFT cube fftshifted for plotting with ``imshow``.

        Returns:
            (torch.complex tensor, of shape ``(nchan, npix, npix)``): the FFT of the image cube, in sky plane format.
        """

        return torch.fft.fftshift(self.vis, dim=(1, 2))


# class ImageCubeOld(nn.Module):
#     r"""
#     A PyTorch layer that provides a parameter set and transformations to model interferometric visibilities.

#     The parameter set is the pixel values of the image cube itself. The transformations are the real fast Fourier transform (RFFT) and band-limited interpolation routines. The pixels are assumed to represent samples of the specific intensity and are given in units of [:math:`\mathrm{Jy}\,\mathrm{arcsec}^{-2}`].

#     All keyword arguments are required unless noted.

#     Args:
#         npix (int): the number of pixels per image side
#         nchan (int): the number of channels in the image
#         cell_size (float): the width of a pixel [arcseconds]
#         cube (torch.double tensor, optional): an image cube to initialize the model with. If None, assumes starting ``cube`` is ``torch.zeros``.
#         pixel_mapping (torch.nn): a PyTorch function mapping the base pixel representation to the cube representation. If `None`, defaults to `torch.nn.Softplus()`.
#     """

#     def __init__(
#         self,
#         npix=None,
#         nchan=None,
#         cell_size=None,
#         cube=None,
#         pixel_mapping=None,
#         **kwargs
#     ):

#         super().__init__()
#         assert npix % 2 == 0, "npix must be even (for now)"
#         self.npix = int(npix)

#         assert cell_size > 0.0, "cell_size must be positive (arcseconds)"
#         self.cell_size = cell_size * arcsec  # [radians]
#         # cell_size is also the differential change in sky angles
#         # dll = dmm = cell_size #[radians]

#         assert nchan > 0, "must have a positive number of channels"
#         self.nchan = int(nchan)

#         img_radius = self.cell_size * (self.npix // 2)  # [radians]

#         # the output spatial frequencies of the RFFT routine (unshifted)
#         self.us = np.fft.rfftfreq(self.npix, d=self.cell_size) * 1e-3  # convert to [kλ]
#         self.vs = np.fft.fftfreq(self.npix, d=self.cell_size) * 1e-3  # convert to [kλ]

#         # the fft-packed versions corresponding to vis
#         self._us_2D, self._vs_2D = np.meshgrid(
#             self.us, self.vs, indexing="xy"
#         )  # cartesian indexing (default)
#         self._qs_2D = np.sqrt(self._us_2D ** 2 + self._vs_2D ** 2)

#         # the normal 2D versions corresponding to vis
#         self.us_2D = np.fft.fftshift(self._us_2D, axes=0)
#         self.vs_2D = np.fft.fftshift(self._vs_2D, axes=0)
#         self.qs_2D = np.fft.fftshift(self._qs_2D, axes=0)

#         if pixel_mapping is None:
#             self.pixel_mapping = torch.nn.Softplus()
#         else:
#             # TODO assert that this is a PyTorch function
#             self.pixel_mapping = pixel_mapping

#         # The ``_cube`` attribute shouldn't really be accessed by the user, since it's naturally
#         # packed in the fftshifted format to make the Fourier transformation easier
#         # and with East pointing right (i.e., RA increasing to the right)
#         # this is contrary to the way astronomers normally plot images, but
#         # is correct for what the FFT expects
#         if cube is None:
#             self._base_cube = nn.Parameter(
#                 torch.full(
#                     (self.nchan, self.npix, self.npix),
#                     fill_value=0.05,
#                     requires_grad=True,
#                     dtype=torch.double,
#                 )
#             )

#         else:
#             # we expect the user to supply an image cube as it looks on the sky
#             # with East pointing to the left. Therefore we will need to
#             # flip the image across the RA dimension
#             # so that the native cube has East (l) increasing with array index
#             # North (m) should already be increasing with array index
#             flipped = torch.flip(cube, (2,))
#             shifted = utils.fftshift(flipped, axes=(1, 2))
#             import warnings

#             warnings.warn(
#                 "Inverse of pixel mapping not yet implemented. Assigning cube to base_cube as is."
#             )
#             self._base_cube = nn.Parameter(shifted)

#         # calculate the image axes corresponding to the shifted _cube
#         # the native _cube is stored as an FFT-shifted version of
#         # a cube with East (l) increasing with array index and North (m) increasing
#         # with array index
#         self._ll = np.flip(
#             np.fft.ifftshift(utils.fftspace(img_radius, self.npix))
#         )  # [radians]
#         self._mm = np.fft.ifftshift(utils.fftspace(img_radius, self.npix))  # [radians]

#         # the image units are Jy/arcsec^2. An extended source with a brightness temperature
#         # of 100 K is about 4 Jy/arcsec^2. These choice of units helps prevent
#         # loss of numerical precision

#         # calculate the gridding correction function to apply to _cube
#         # evaluated over the (preshifted) _ll and _mm coordinates
#         self.corrfun = torch.tensor(spheroidal_gridding.corrfun_mat(self._ll, self._mm))

#         self.precached = False

#     def precache_interpolation(self, dataset):
#         """
#         Cache the interpolation matrices used to interpolate the output from the RFFT to the measured :math:`(u,v)` points. This is only applicable if the dataset has not been pre-gridded, and will be run automatically upon the first call to :meth:`mpol.ImageCube.forward`.

#         Stores the attributes ``C_res`` and ``C_ims``, which are lists of sparse interpolation matrices corresponding to each channel.

#         Args:
#             dataset (UVDataset): a UVDataset containing the :math:`(u,v)` sampling points of the observation.

#         Returns:
#             None


#         """

#         max_baseline = torch.max(
#             torch.abs(torch.cat([dataset.uu, dataset.vv]))
#         )  # klambda

#         # check that the pixel scale is sufficiently small to sample
#         # the frequency corresponding to the largest baseline of the
#         # dataset (in klambda)
#         assert max_baseline < (
#             1e-3 / (2 * self.cell_size)
#         ), "Image cell size is too coarse to represent the largest spatial frequency sampled by the dataset. Make a finer image by decreasing cell_size. You may also need to increase npix to make sure the image remains wide enough to capture all of the emission and avoid aliasing."

#         # calculate the interpolation matrices at the datapoints
#         # the .detach().cpu() is to enable the numpy conversion even after transferred to GPU
#         uu = dataset.uu.detach().cpu().numpy()
#         vv = dataset.vv.detach().cpu().numpy()
#         self.C_res = []
#         self.C_ims = []
#         for i in range(self.nchan):
#             C_re, C_im = spheroidal_gridding.calc_matrices(
#                 uu[i], vv[i], self.us, self.vs
#             )
#             C_shape = C_re.shape

#             # make these torch sparse tensors
#             i_re = torch.LongTensor([C_re.row, C_re.col])
#             v_re = torch.DoubleTensor(C_re.data)
#             C_re = torch.sparse.DoubleTensor(i_re, v_re, torch.Size(C_shape))
#             self.C_res.append(C_re)

#             i_im = torch.LongTensor([C_im.row, C_im.col])
#             v_im = torch.DoubleTensor(C_im.data)
#             C_im = torch.sparse.DoubleTensor(i_im, v_im, torch.Size(C_shape))
#             self.C_ims.append(C_im)

#         self.precached = True

#     def forward(self, dataset):
#         r"""
#         Compute the model visibilities at the :math:`(u, v)` locations of the dataset.

#         Args:
#             dataset (UVDataset): the dataset to forward model.

#         Returns:
#             (torch.double, torch.double): a 2-tuple of the :math:`\Re` and :math:`\Im` model values.
#         """

#         if dataset.gridded:
#             # re, im output will always be 1D
#             assert (
#                 dataset.npix == self.npix
#             ), "Pre-gridded npix is different than model npix"
#             assert (
#                 dataset.cell_size == self.cell_size
#             ), "Pre-gridded cell_size is different than model cell_size."

#             # convert the image to Jy/ster and perform the RFFT
#             # the self.cell_size prefactor (in radians) is to obtain the correct output units
#             # since it needs to correct for the spacing of the input grid.
#             # See MPoL documentation and/or TMS Eqn A8.18 for more information.
#             # Alternatively we could send the rfft routine _cube in its native Jy/arcsec^2
#             # and then multiply the result by cell_size (in units of arcsec).
#             # It seemed easiest to do it this way where we keep things in radians.
#             self._vis = self.cell_size ** 2 * torch.rfft(
#                 self._cube / arcsec ** 2, signal_ndim=2
#             )

#             # torch delivers the real and imag components separately
#             vis_re = self._vis[:, :, :, 0]
#             vis_im = self._vis[:, :, :, 1]

#             # grid mask is a (nchan, npix, npix//2 + 1) size boolean array
#             re = vis_re.masked_select(dataset.grid_mask)
#             im = vis_im.masked_select(dataset.grid_mask)

#         else:
#             # re, im output will always be 2D (nchan, nvis)
#             # test to see if the interpolation is pre-cached
#             if not self.precached:
#                 # this routine checks that the maxbaseline is contained within the grid.
#                 self.precache_interpolation(dataset)

#             # TODO: does the corrfun broadcast correctly across the cube?
#             self._vis = self.cell_size ** 2 * torch.rfft(
#                 self._cube * self.corrfun / arcsec ** 2, signal_ndim=2
#             )

#             # torch delivers the real and imag components separately
#             vis_re = self._vis[:, :, :, 0]
#             vis_im = self._vis[:, :, :, 1]

#             # reshape into (nchan, -1, 1) vector format so we can do matrix product
#             vr = torch.reshape(vis_re, (self.nchan, -1, 1))
#             vi = torch.reshape(vis_im, (self.nchan, -1, 1))

#             res = []
#             ims = []
#             # sample the FFT using the sparse matrices
#             # the output of mm is a (nvis, 1) dimension tensor
#             for i in range(self.nchan):
#                 res.append(torch.sparse.mm(self.C_res[i], vr[i]))
#                 ims.append(torch.sparse.mm(self.C_ims[i], vi[i]))

#             # concatenate to a single (nchan, nvis) tensor
#             re = torch.transpose(torch.cat(res, dim=1), 0, 1)
#             im = torch.transpose(torch.cat(ims, dim=1), 0, 1)

#         return re, im

#     @property
#     def _cube(self):
#         """
#         The shifted image cube.
#         """

#         return self.pixel_mapping(self._base_cube)

#     @property
#     def cube(self):
#         """
#         The image cube.

#         Returns:
#             torch.double : image cube of shape ``(nchan, npix, npix)``

#         """
#         # fftshift the image cube to the correct quadrants
#         shifted = utils.fftshift(self._cube, axes=(1, 2))
#         # flip so that east points left
#         flipped = torch.flip(shifted, (2,))
#         return flipped

#     @property
#     def extent(self):
#         r"""
#         The extent 4-tuple (in arcsec) to assign relative image coordinates (:math:`\Delta \alpha \cos \delta`,  :math:`\Delta \delta`) with matplotlib imshow. Assumes ``origin="lower"``.

#         Returns:
#             4-tuple: extent
#         """
#         low = np.min(self._ll) / arcsec - 0.5 * self.cell_size  # [arcseconds]
#         high = np.max(self._ll) / arcsec + 0.5 * self.cell_size  # [arcseconds]

#         return [high, low, low, high]

#     @property
#     def vis(self):
#         r"""
#         The visibility RFFT cube fftshifted for plotting with ``imshow`` (the v coordinate goes from -ve to +ve).

#         Returns:
#             torch.double: visibility cube
#         """

#         return utils.fftshift(self._vis, axes=(1,))

#     @property
#     def psd(self):
#         r"""
#         The power spectral density of the cube, fftshifted for plotting. (The v coordinate goes from -ve to +ve).

#         Returns:
#             torch.double: power spectral density cube
#         """

#         vis_re = self.vis[:, :, :, 0]
#         vis_im = self.vis[:, :, :, 1]

#         psd = vis_re ** 2 + vis_im ** 2

#         return psd

#     @property
#     def vis_extent(self):
#         r"""
#         The `imshow` ``extent`` argument corresponding to `vis_cube` when plotted with ``origin="lower"``. The :math:`(u, v)` coordinates.

#         Returns:
#             4-tuple: extent
#         """
#         du = 1 / (self.npix * self.cell_size) * 1e-3  # klambda
#         left = np.min(self.us) - 0.5 * du
#         right = np.max(self.us) + 0.5 * du
#         bottom = np.min(self.vs) - 0.5 * du
#         top = np.max(self.vs) + 0.5 * du

#         return [left, right, bottom, top]

#     def to_FITS(self, fname="cube.fits", overwrite=False, header_kwargs=None):
#         """
#         Export the image cube to a FITS file.

#         Args:
#             fname (str): the name of the FITS file to export to.
#             overwrite (bool): if the file already exists, overwrite?
#             header_kwargs (dict): Extra keyword arguments to write to the FITS header.

#         Returns:
#             None
#         """

#         try:
#             from astropy.io import fits
#             from astropy import wcs
#         except ImportError:
#             print(
#                 "Please install the astropy package to use FITS export functionality."
#             )

#         w = wcs.WCS(naxis=2)

#         w.wcs.crpix = np.array([1, 1])
#         w.wcs.cdelt = (
#             np.array([self.cell_size, self.cell_size]) * 180.0 / np.pi
#         )  # decimal degrees
#         w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

#         header = w.to_header()

#         # add in the kwargs to the header
#         if header_kwargs is not None:
#             for k, v in header_kwargs.items():
#                 header[k] = v

#         hdu = fits.PrimaryHDU(self.cube.detach().cpu().numpy(), header=header)

#         hdul = fits.HDUList([hdu])
#         hdul.writeto(fname, overwrite=overwrite)

#         hdul.close()
