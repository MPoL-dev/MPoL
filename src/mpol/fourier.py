r"""The ``fourier`` module provides the core functionality of MPoL via :class:`mpol.fourier.FourierCube`."""

import numpy as np
import torch
import torch.fft  # to avoid conflicts with old torch.fft *function*
import torchkbnufft
from torch import nn

from . import utils
from .coordinates import GridCoords
from .gridding import _setup_coords


class FourierCube(nn.Module):
    r"""
    This layer performs the FFT of an ImageCube and stores the corresponding dense FFT output as a cube. If you are using this layer in a forward-modeling RML workflow, because the FFT of the model is essentially stored as a grid, you will need to make the loss function calculation using a gridded loss function (e.g., :func:`mpol.losses.nll_gridded`) and a gridded dataset (e.g., :class:`mpol.datasets.GriddedDataset`).

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
        Perform the FFT of the image cube on each channel.

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
        self.vis = self.coords.cell_size**2 * torch.fft.fftn(cube, dim=(1, 2))

        return self.vis

    @property
    def ground_vis(self):
        r"""
        The visibility cube in ground format cube fftshifted for plotting with ``imshow``.

        Returns:
            (torch.complex tensor, of shape ``(nchan, npix, npix)``): the FFT of the image cube, in sky plane format.
        """

        return utils.packed_cube_to_ground_cube(self.vis)

    @property
    def ground_amp(self):
        r"""
        The amplitude of the cube, arranged in unpacked format corresponding to the FFT of the sky_cube. Array dimensions for plotting given by ``self.coords.vis_ext``.

        Returns:
            torch.double : 3D amplitude cube of shape ``(nchan, npix, npix)``
        """
        return torch.abs(self.ground_vis)

    @property
    def ground_phase(self):
        r"""
        The phase of the cube, arranged in unpacked format corresponding to the FFT of the sky_cube. Array dimensions for plotting given by ``self.coords.vis_ext``.

        Returns:
            torch.double : 3D phase cube of shape ``(nchan, npix, npix)``
        """
        return torch.angle(self.ground_vis)


class NuFFT(nn.Module):
    r"""
    This layer translates input from an :class:`mpol.images.ImageCube` directly to loose, ungridded samples of the Fourier plane, directly corresponding to the :math:`u,v` locations of the data. This layer is different than a :class:`mpol.Fourier.FourierCube` in that, rather than producing the dense cube-like output from an FFT routine, it utilizes the non-uniform FFT or 'NuFFT' to interpolate directly to discrete :math:`u,v` locations that need not correspond to grid cell centers. This is implemented using the KbNufft routines of the `TorchKbNufft <https://torchkbnufft.readthedocs.io/en/stable/index.html>`_ package.

    **Dimensionality**: One consideration when using this layer is the dimensionality of your image and your visibility samples. If your image has multiple channels (``nchan > 1``), there is the possibility that the :math:`u,v` sample locations corresponding to each channel may be different. In ALMA/VLA applications, this can arise when continuum observations are taken over significant bandwidth, since the spatial frequency sampled by any pair of antennas is wavelength-dependent

    .. math::

        u = \frac{D}{\lambda},

    where :math:`D` is the projected baseline (measured in, say, meters) and :math:`\lambda` is the observing wavelength. In this application, the image-plane model could be the same for each channel, or it may vary with channel (necessary if the spectral slope of the source is significant).

    On the other hand, with spectral line observations it will usually be the case that the total bandwidth of the observations is small enough such that the :math:`u,v` sample locations could be considered as the same for each channel. In spectral line applications, the image-plane model usually varies substantially with each channel.

    This layer will determine whether the spatial frequencies are treated as constant based upon the dimensionality of the ``uu`` and ``vv`` input arguments.

    * If ``uu`` and ``vv`` have a shape of (``nvis``), then it will be assumed that the spatial frequencies can be treated as constant with channel (and will invoke parallelization across the image cube ``nchan`` dimension using the 'coil' dimension of the TorchKbNufft package).
    * If the ``uu`` and ``vv`` have a shape of (``nchan, nvis``), then it will be assumed that the spatial frequencies are different for each channel, and the spatial frequencies provided for each channel will be used (and will invoke parallelization across the image cube ``nchan`` dimension using the 'batch' dimension of the TorchKbNufft package).

    Note that there is no straightforward, computationally efficient way to proceed if there are a different number of spatial frequencies for each channel. The best approach is likely to construct ``uu`` and ``vv`` arrays that have a shape of (``nchan, nvis``), such that all channels are padded with bogus :math:`u,v` points to have the same length ``nvis``, and you create a boolean mask to keep track of which points are valid. Then, when this routine returns data points of shape (``nchan, nvis``), you can use that boolean mask to select only the valid :math:`u,v` points points.

    **Interpolation mode**: You may choose the type of interpolation mode that KbNufft uses under the hood by changing the boolean value of ``sparse_matrices``. For repeated evaluations of this layer (as might exist within an optimization loop), ``sparse_matrices=True`` is likely to be the more accurate and faster choice. If ``sparse_matrices=False``, this routine will use the default table-based interpolation of TorchKbNufft.

    Args:
        cell_size (float): the width of an image-plane pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        uu (np.array): a length ``nvis`` array (not including Hermitian pairs) of the u (East-West) spatial frequency coordinate [klambda]
        vv (np.array): a length ``nvis`` array (not including Hermitian pairs) of the v (North-South) spatial frequency coordinate [klambda]

    """

    def __init__(
        self,
        cell_size=None,
        npix=None,
        coords=None,
        uu=None,
        vv=None,
        sparse_matrices=True,
    ):

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

        # initialize the non-uniform FFT object
        self.nufft_ob = torchkbnufft.KbNufft(
            im_size=(self.coords.npix, self.coords.npix)
        )

        if (uu is not None) and (vv is not None):
            self.k_traj = self._assemble_ktraj(uu, vv)

    def _klambda_to_radpix(self, klambda):
        """Convert a spatial frequency in units of klambda to radians/pixel.

        Args:
            klambda (float): spatial frequency in units of kilolambda
            cell_size (float): the size of a pixel in units of arcsec
        """

        # cycles per sky radian
        u_lam = klambda * 1e3  # [lambda, or cycles/radian]

        # radians per sky radian
        u_rad_per_rad = u_lam * 2 * np.pi  # [radians / sky radian]

        # size of pixel in radians
        # self.coords.dl  # [sky radians/pixel]

        # radians per pixel
        u_rad_per_pix = u_rad_per_rad * self.coords.dl  # [radians / pixel]

        return u_rad_per_pix

    def _assemble_ktraj(self, uu, vv):
        r"""
        Convert a series of :math:`u, v` coordinates into a k-trajectory vector for the torchkbnufft routines.

        Args:
            uu (numpy array): u (East-West) spatial frequency coordinate [klambda]
            vv (numpy array): v (North-South) spatial frequency coordinate [klambda]
        """

        uu_radpix = self._klambda_to_radpix(uu)
        vv_radpix = self._klambda_to_radpix(vv)

        # k-trajectory needs to be packed the way the image is packed (y,x), so
        # the trajectory needs to be packed (v, u)
        k_traj = torch.tensor([vv_radpix, uu_radpix])

        return k_traj

    def forward(self, cube, uu=None, vv=None):
        r"""
        Perform the FFT of the image cube for each channel and interpolate to the ``uu`` and ``vv`` points set at layer initialization.

        Args:
            cube (torch.double tensor): of shape ``(nchan, npix, npix)``). A prepacked image cube, for example, from :meth:`mpol.images.ImageCube.forward`

        Returns:
            torch.complex tensor: of shape ``(nchan, nvis)``, Fourier samples evaluate from the input image cube at the locations of the ``uu``, ``vv`` points set at layer initialization.
        """

        if (uu is not None) and (vv is not None):
            k_traj = self._assemble_ktraj(uu, vv)
        else:
            k_traj = self.k_traj

        # "unpack" the cube, but leave it flipped
        shifted = torch.fft.fftshift(cube, dim=(1, 2))

        # convert the cube to an imaginary value
        complexed = shifted.type(torch.complex128)

        # expand the cube to include a batch dimension
        expanded = complexed.unsqueeze(0)
        # now [1, nchan, npix, npix] shape

        # send this through the object
        output = self.coords.cell_size**2 * self.nufft_ob(expanded, k_traj)
        # output is shape [1, nchan, ntraj]

        # remove the batch dimension
        return output[0]
