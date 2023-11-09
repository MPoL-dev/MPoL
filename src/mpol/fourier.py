r"""The ``fourier`` module provides the core functionality of MPoL via :class:`mpol.fourier.FourierCube`."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.fft  # to avoid conflicts with old torch.fft *function*
import torchkbnufft
from numpy import complexfloating, floating
from numpy.typing import NDArray
from torch import nn

from mpol.exceptions import DimensionMismatchError
from mpol.images import ImageCube
from mpol.protocols import MPoLModel

from . import utils
from .coordinates import GridCoords


class FourierCube(nn.Module):
    r"""
    This layer performs the FFT of an ImageCube and stores the corresponding dense FFT output as a cube. If you are using this layer in a forward-modeling RML workflow, because the FFT of the model is essentially stored as a grid, you will need to make the loss function calculation using a gridded loss function (e.g., :func:`mpol.losses.nll_gridded`) and a gridded dataset (e.g., :class:`mpol.datasets.GriddedDataset`).

    Args:
        coords (GridCoords): an object already instantiated from the GridCoords class.
        persistent_vis (Boolean): should the visibility cube be stored as part of the modules `state_dict`? If `True`, the state of the UV grid will be stored. It is recommended to use `False` for most applications, since the visibility cube will rarely be a direct parameter of the model.

    """

    def __init__(self, coords: GridCoords, persistent_vis: bool = False):
        super().__init__()

        # TODO: Is this comment relevant? There was no nchan instantiation
        # before
        # ---
        # we don't want to bother with the nchan argument here, so
        # we don't use the convenience method _setup_coords
        # and just do it manually

        self.coords = coords

        self.register_buffer("vis", None, persistent=persistent_vis)
        self.vis: torch.Tensor

    @classmethod
    def from_image_properties(
        cls, cell_size: float, npix: int, persistent_vis: bool = False
    ) -> FourierCube:
        """
        Alternative method for instantiating a FourierCube from ``cell_size`` and ``npix``

        Args:
            cell_size (float): the width of an image-plane pixel [arcseconds]
            npix (int): the number of pixels per image side
            persistent_vis (Boolean): should the visibility cube be stored as part of the modules `state_dict`? If `True`, the state of the UV grid will be stored. It is recommended to use `False` for most applications, since the visibility cube will rarely be a direct parameter of the model.

        Returns:
            instantiated :class:`mpol.fourier.FourierCube` object.
        """
        coords = GridCoords(cell_size, npix)
        return cls(coords, persistent_vis)

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
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
    def ground_vis(self) -> torch.Tensor:
        r"""
        The visibility cube in ground format cube fftshifted for plotting with ``imshow``.

        Returns:
            (torch.complex tensor, of shape ``(nchan, npix, npix)``): the FFT of the image cube, in sky plane format.
        """

        return utils.packed_cube_to_ground_cube(self.vis)

    @property
    def ground_amp(self) -> torch.Tensor:
        r"""
        The amplitude of the cube, arranged in unpacked format corresponding to the FFT of the sky_cube. Array dimensions for plotting given by ``self.coords.vis_ext``.

        Returns:
            torch.double : 3D amplitude cube of shape ``(nchan, npix, npix)``
        """
        return torch.abs(self.ground_vis)

    @property
    def ground_phase(self) -> torch.Tensor:
        r"""
        The phase of the cube, arranged in unpacked format corresponding to the FFT of the sky_cube. Array dimensions for plotting given by ``self.coords.vis_ext``.

        Returns:
            torch.double : 3D phase cube of shape ``(nchan, npix, npix)``
        """
        return torch.angle(self.ground_vis)


def safe_baseline_constant_meters(
    uu: NDArray[floating[Any]],
    vv: NDArray[floating[Any]],
    freqs: NDArray[floating[Any]],
    coords: GridCoords,
    uv_cell_frac: float = 0.05,
) -> bool:
    r"""
    This routine determines whether the baselines can safely be assumed to be constant with channel when they converted from meters to units of kilolambda.

    The antenna baselines *are* the same as a function of channel when they are measured in physical distance units, such as meters. However, when these baselines are converted to spatial frequency units, via

    .. math::

        u = \frac{D}{\lambda},

    it's possible that the :math:`u` and :math:`v` values of each channel are significantly different if the :math:`\lambda` values of each channel are significantly different. This routine evaluates whether the maximum change in :math:`u` or :math:`v` across channels (when represented in kilolambda) is smaller than some threshold value, calculated as the fraction of a :math:`u,v` cell defined by ``coords``.

    If this function returns ``True``, then it would be safe to proceed with parallelization in the :class:`mpol.fourier.NuFFT` layer via the coil dimension.

    Args:
        uu (1D np.array): a 1D array of length ``nvis`` array of the u (East-West) spatial frequency coordinate in units of [m]
        vv (1D np.array): a 1D array of length ``nvis`` array of the v (North-South) spatial frequency coordinate in units of [m]
        freqs (1D np.array): a 1D array of length ``nchan`` of the channel frequencies, in units of [Hz].
        coords: a :class:`mpol.coordinates.GridCoords` object which represents the image and uv-grid dimensions.
        uv_cell_frac (float): the maximum threshold for a change in :math:`u` or :math:`v` spatial frequency across channels, measured as a fraction of the :math:`u,v` cell defined by ``coords``.

    Returns:
        boolean: `True` if it is safe to assume that the baselines are constant with channel (at a tolerance of ``uv_cell_frac``.) Otherwise returns `False`.
    """

    # broadcast and convert baselines to kilolambda across channel
    uu, vv = utils.broadcast_and_convert_baselines(uu, vv, freqs)
    # should be (nchan, nvis) arrays

    # convert uv_cell_frac to a kilolambda threshold
    delta_uv = uv_cell_frac * coords.du  # [klambda]

    # find maximum change in baseline across channel
    # concatenate arrays to save steps
    uv = np.array([uu, vv])  # (2, nchan, nvis) arrays

    # find max - min along channel axis
    uv_min = uv.min(axis=1)
    uv_max = uv.max(axis=1)
    uv_diff = uv_max - uv_min

    # find maximum of that
    max_diff: float = uv_diff.max()

    # compare to uv_cell_frac
    return max_diff < delta_uv


def safe_baseline_constant_kilolambda(
    uu: NDArray[floating[Any]],
    vv: NDArray[floating[Any]],
    coords: GridCoords,
    uv_cell_frac: float = 0.05,
) -> bool:
    r"""
    This routine determines whether the baselines can safely be assumed to be constant with channel, when the are represented in units of kilolambda.

    Compared to :class:`mpol.fourier.safe_baseline_constant_meters`, this function works with multidimensional arrays of ``uu`` and ``vv`` that are shape (nchan, nvis) and have units of kilolambda.

    If this routine returns True, then it should be safe for the user to either average the baselines across channel or simply choose a single, representative channel. This would enable parallelization in the {class}`mpol.fourier.NuFFT` via the coil dimension.

    Args:
        uu (1D np.array): a 1D array of length ``nvis`` array of the u (East-West) spatial frequency coordinate in units of [m]
        vv (1D np.array): a 1D array of length ``nvis`` array of the v (North-South) spatial frequency coordinate in units of [m]
        freqs (1D np.array): a 1D array of length ``nchan`` of the channel frequencies, in units of [Hz].
        coords: a :class:`mpol.coordinates.GridCoords` object which represents the image and uv-grid dimensions.
        uv_cell_frac (float): the maximum threshold for a change in :math:`u` or :math:`v` spatial frequency across channels, measured as a fraction of the :math:`u,v` cell defined by ``coords``.

    Returns:
        boolean: `True` if it is safe to assume that the baselines are constant with channel (at a tolerance of ``uv_cell_frac``.) Otherwise returns `False`.

    """
    # convert uv_cell_frac to a kilolambda threshold
    delta_uv = uv_cell_frac * coords.du  # [klambda]

    # find maximum change in baseline across channel
    # concatenate arrays to save steps
    uv = np.array([uu, vv])  # (2, nchan, nvis) arrays

    # find max - min along channel axis
    uv_min = uv.min(axis=1)
    uv_max = uv.max(axis=1)
    uv_diff = uv_max - uv_min

    # find maximum of that
    max_diff: float = uv_diff.max()

    # compare to uv_cell_frac
    return max_diff < delta_uv


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

    **Interpolation mode**: You may choose the type of interpolation mode that KbNufft uses under the hood by changing the boolean value of ``sparse_matrices``. For repeated evaluations of this layer (as might exist within an optimization loop), ``sparse_matrices=True`` is likely to be the more accurate and faster choice. If ``sparse_matrices=False``, this routine will use the default table-based interpolation of TorchKbNufft. Note that as of TorchKbNuFFT version 1.4.0, sparse matrices are not yet available when parallelizing using the 'batch' dimension --- this will result in a warning.

    Args:
        cell_size (float): the width of an image-plane pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        nchan (int): the number of channels in the :class:`mpol.images.ImageCube`. Default = 1.
        uu (np.array): a length ``nvis`` array (not including Hermitian pairs) of the u (East-West) spatial frequency coordinate [klambda]
        vv (np.array): a length ``nvis`` array (not including Hermitian pairs) of the v (North-South) spatial frequency coordinate [klambda]

    """

    def __init__(
        self,
        coords: GridCoords,
        uu: NDArray[floating[Any]],
        vv: NDArray[floating[Any]],
        nchan: int = 1,
        sparse_matrices: bool = True,
    ):
        super().__init__()

        if not (same_uv := uu.ndim == 1 and vv.ndim == 1) and sparse_matrices:
            import warnings

            warnings.warn(
                "Provided uu and vv arrays are multi-dimensional, suggesting an "
                "intent to parallelize using the 'batch' dimension. This feature "
                "is not yet available in TorchKbNuFFT v1.4.0 with sparse matrix "
                "interpolation (sparse_matrices=True), therefore we are proceeding "
                "with table interpolation (sparse_matrices=False).",
                category=RuntimeWarning,
            )
            sparse_matrices = False
            self.interp_mat = None

        self.coords = coords
        self.nchan = nchan
        self.sparse_matrices = sparse_matrices
        self.same_uv = same_uv

        # initialize the non-uniform FFT object
        self.nufft_ob = torchkbnufft.KbNufft(
            im_size=(self.coords.npix, self.coords.npix)
        )

        self.register_buffer("k_traj", self._assemble_ktraj(uu, vv))
        self.k_traj: torch.Tensor

        if self.sparse_matrices:
            # precompute the sparse interpolation matrices
            real_interp_mat, imag_interp_mat = torchkbnufft.calc_tensor_spmatrix(
                self.k_traj, im_size=(self.coords.npix, self.coords.npix)
            )
            self.register_buffer("real_interp_mat", real_interp_mat)
            self.register_buffer("imag_interp_mat", imag_interp_mat)
            self.real_interp_mat: torch.Tensor
            self.imag_interp_mat: torch.Tensor

    @classmethod
    def from_image_properties(
        cls,
        cell_size: float,
        npix: int,
        uu: NDArray[floating[Any]],
        vv: NDArray[floating[Any]],
        nchan: int = 1,
        sparse_matrices: bool = True,
    ) -> NuFFT:
        coords = GridCoords(cell_size, npix)
        return cls(coords, uu, vv, nchan, sparse_matrices)

    def _klambda_to_radpix(
        self, klambda: float | NDArray[floating[Any]]
    ) -> float | NDArray[floating[Any]]:
        """Convert a spatial frequency in units of klambda to 'radians/sky pixel,' using the pixel cell_size provided by ``self.coords.dl``.

        These concepts can be a little confusing because there are two angular measures at play.

        1. The first is the normal angular sky coordinate, normally measured in arcseconds for typical sources observed by ALMA or the VLA. Arcseconds, being an angular coordinate, can equivalently be expressed in units of radians. To avoid confusion, we will call this angular measurement 'sky radians.' Alternatively, for a given image grid, this same sky coordinate could be expressed in units of sky pixels.
        2. The second is the spatial frequency of some image-plane function, :math:`I_\nu(l,m)`, which we could quote in units of 'cycles per arcsecond' or 'cycles per sky pixel,' for example. With a radio interferometer, spatial frequencies are typically quoted in units of the observing wavelength, i.e., lambda or kilo-lambda. If the field of view of the image is small, thanks to the small-angle approximation, units of lambda are directly equivalent to 'cycles per sky radian.' The second angular measure comes about when converting the spatial frequency from a linear measure of frequency 'cycles per sky radian' to an angular measure of frequency 'radians per sky radian' or 'radians per sky pixel.'

        The TorchKbNufft package expects k-trajectory vectors in units of 'radians per sky pixel.' This routine helps convert spatial frequencies from their default unit (kilolambda) into 'radians per sky pixel' using the pixel cell_size as provided by ``self.coords.dl``.

        Args:
            klambda (float): spatial frequency in units of kilolambda

        Returns:
            float: spatial frequency measured in units of radian per sky pixel
        """

        # convert from kilolambda to cycles per sky radian
        u_lam = klambda * 1e3  # [lambda, or cycles/radian]

        # convert from 'cycles per sky radian' to 'radians per sky radian'
        u_rad_per_rad = u_lam * 2 * np.pi  # [radians / sky radian]

        # size of pixel in radians
        # self.coords.dl  # [sky radians/pixel]

        # convert from 'radians per sky radian' to 'radians per sky pixel'
        u_rad_per_pix = u_rad_per_rad * self.coords.dl  # [radians / pixel]

        return u_rad_per_pix

    def _assemble_ktraj(
        self, uu: NDArray[floating[Any]], vv: NDArray[floating[Any]]
    ) -> torch.Tensor:
        r"""
        This routine converts a series of :math:`u, v` coordinates into a k-trajectory vector for the torchkbnufft routines. The dimensionality of the k-trajectory vector will influence how TorchKbNufft will perform the operations.

        * If ``uu`` and ``vv`` have a 1D shape of (``nvis``), then it will be assumed that the spatial frequencies can be treated as constant with channel. This will result in a ``k_traj`` vector that has shape (``2, nvis``), such that parallelization will be across the image cube ``nchan`` dimension using the 'coil' dimension of the TorchKbNufft package.
        * If the ``uu`` and ``vv`` have a 2D shape of (``nchan, nvis``), then it will be assumed that the spatial frequencies are different for each channel, and the spatial frequencies provided for each channel will be used. This will result in a ``k_traj`` vector that has shape (``nchan, 2, nvis``), such that parallelization will be across the image cube ``nchan`` dimension using the 'batch' dimension of the TorchKbNufft package.

        Args:
            uu (1D or 2D numpy array): u (East-West) spatial frequency coordinate [klambda]
            vv (1D or 2D numpy array): v (North-South) spatial frequency coordinate [klambda]

        Returns:
            k_traj (torch tensor): a k-trajectory vector with shape
        """

        uu_radpix = self._klambda_to_radpix(uu)
        vv_radpix = self._klambda_to_radpix(vv)

        assert isinstance(uu_radpix, np.ndarray)
        assert isinstance(vv_radpix, np.ndarray)

        # if uu and vv are 1D dimension, then we can assume that we will parallelize across the coil dimension.
        # otherwise, we assume that we will parallelize across the batch dimension.
        if self.same_uv:
            # k-trajectory needs to be packed the way the image is packed (y,x), so
            # the trajectory needs to be packed (v, u)
            # if TorchKbNufft receives a k-traj tensor of shape (2, nvis), it will parallelize across the coil dimension, assuming
            # that the k-traj is the same for all coils/channels.
            # interim convert to numpy array because of torch warning about speed
            k_traj = torch.tensor(np.array([vv_radpix, uu_radpix]))
            return k_traj

        # in this case, we are given two tensors of shape (nchan, nvis)
        # first, augment each tensor individually to create a (nbatch, 1, nvis) tensor
        # then, concatenate the tensors along the axis=1 dimension.

        if uu_radpix.shape[0] != self.nchan:
            raise DimensionMismatchError(
                f"nchan of uu ({uu_radpix.shape[0]}) is more than one but different than that used to initialize the NuFFT layer ({self.nchan})"
            )

        if vv_radpix.shape[0] != self.nchan:
            raise DimensionMismatchError(
                f"nchan of vv ({vv_radpix.shape[0]}) is more than one but different than that used to initialize the NuFFT layer ({self.nchan})"
            )

        uu_radpix_aug = torch.unsqueeze(torch.tensor(uu_radpix), 1)
        vv_radpix_aug = torch.unsqueeze(torch.tensor(vv_radpix), 1)

        # interim convert to numpy array because of torch warning about speed
        k_traj = torch.cat([vv_radpix_aug, uu_radpix_aug], dim=1)
        # if TorchKbNufft receives a k-traj tensor of shape (nbatch, 2, nvis), it will parallelize across the batch dimension

        return k_traj

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        r"""
        Perform the FFT of the image cube for each channel and interpolate to the ``uu`` and ``vv`` points set at layer initialization. This call should automatically take the best parallelization option as indicated by the shape of the ``uu`` and ``vv`` points.

        Args:
            cube (torch.double tensor): of shape ``(nchan, npix, npix)``). The cube should be a "prepacked" image cube, for example, from :meth:`mpol.images.ImageCube.forward`

        Returns:
            torch.complex tensor: of shape ``(nchan, nvis)``, Fourier samples evaluated corresponding to the ``uu``, ``vv`` points set at initialization.
        """

        # make sure that the nchan assumptions for the ImageCube and the NuFFT
        # setup are the same
        if cube.size(0) != self.nchan:
            raise DimensionMismatchError(
                f"nchan of ImageCube ({cube.size(0)}) is different than that used to initialize NuFFT layer ({self.nchan})"
            )

        # "unpack" the cube, but leave it flipped
        # NuFFT routine expects a "normal" cube, not an fftshifted one
        shifted = torch.fft.fftshift(cube, dim=(1, 2))

        # convert the cube to a complex type, since this is required by TorchKbNufft
        complexed = shifted.type(torch.complex128)

        # Consider how the similarity of the spatial frequency samples should be
        # treated. We already took care of this on the k_traj side, since we set
        # the shapes. But this also needs to be taken care of on the image side.
        #   * If we plan to parallelize with the coil dimension, then we need an
        #     image with shape (1, nchan, npix, npix).
        #   * If we plan to parallelize using the batch dimension, then we need
        #     an image with shape (nchan, 1, npix, npix).

        if self.same_uv:
            # we want to unsqueeze/squeeze at dim=0 to parallelize over the coil
            # dimension
            # unsquezee shape: [1, nchan, npix, npix]
            altered_dimension = 0
        else:
            # we want to unsqueeze/squeeze at dim=1 to parallelize over the
            # batch dimension
            # unsquezee shape: [nchan, 1, npix, npix]
            altered_dimension = 1

        expanded = complexed.unsqueeze(altered_dimension)

        # torchkbnufft uses a [nbatch, ncoil, npix, npix] scheme
        output: torch.Tensor = self.coords.cell_size**2 * self.nufft_ob(
            expanded,
            self.k_traj,
            interp_mats=(
                (self.real_interp_mat, self.imag_interp_mat)
                if self.sparse_matrices
                else None
            ),
        )

        # squeezed shape: [nchan, npix, npix]
        output = torch.squeeze(output, dim=altered_dimension)

        return output


def make_fake_data(
    image_cube: ImageCube,
    uu: NDArray[floating[Any]],
    vv: NDArray[floating[Any]],
    weight: NDArray[floating[Any]],
) -> tuple[NDArray[complexfloating[Any, Any]], ...]:
    r"""
    Create a fake dataset from a supplied :class:`mpol.images.ImageCube`. See :ref:`mock-dataset-label` for more details on how to prepare a generic image for use in an :class:`~mpol.images.ImageCube`.

    The provided visibilities can be 1d for a single continuum channel, or 2d for image cube. If 1d, visibilities will be converted to 2d arrays of shape ``(1, nvis)``.

    Args:
        imageCube (:class:`~mpol.images.ImageCube`): the image layer to put into a fake dataset
        uu (numpy array): array of u spatial frequency coordinates, not including Hermitian pairs. Units of [:math:`\mathrm{k}\lambda`]
        vv (numpy array): array of v spatial frequency coordinates, not including Hermitian pairs. Units of [:math:`\mathrm{k}\lambda`]
        weight (2d numpy array): length array of thermal weights :math:`w_i = 1/\sigma_i^2`. Units of [:math:`1/\mathrm{Jy}^2`]

    Returns:
        (2-tuple): a two tuple of the fake data. The first array is the mock dataset including noise, the second array is the mock dataset without added noise.
    """

    # instantiate a NuFFT object based on the ImageCube
    # OK if uu shape (nvis,)
    nufft = NuFFT(coords=image_cube.coords, nchan=image_cube.nchan, uu=uu, vv=vv)

    # make into a multi-channel dataset, even if only a single-channel provided
    if uu.ndim == 1:
        uu = np.atleast_2d(uu)
        vv = np.atleast_2d(vv)
        weight = np.atleast_2d(weight)

    # carry it forward to the visibilities, which will be (nchan, nvis)
    vis_noiseless: NDArray[complexfloating[Any, Any]]
    vis_noiseless = nufft(image_cube()).detach().numpy()

    # generate complex noise
    sigma = 1 / np.sqrt(weight)
    noise = np.random.normal(
        loc=0, scale=sigma, size=uu.shape
    ) + 1.0j * np.random.normal(loc=0, scale=sigma, size=uu.shape)

    # add to data
    vis_noise = vis_noiseless + noise

    return vis_noise, vis_noiseless


def get_vis_residuals(
    model: MPoLModel,
    u_true: NDArray[floating[Any]],
    v_true: NDArray[floating[Any]],
    V_true: NDArray[complexfloating[Any, Any]],
    channel: int = 0,
) -> NDArray[complexfloating[Any, Any]]:
    r"""
    Use `mpol.fourier.NuFFT` to get residuals between gridded `model` and loose
    (ungridded) data visiblities at data (u, v) coordinates

    Parameters
    ----------
    model : `torch.nn.Module` object
        Instance of the `mpol.precomposed.SimpleNet` class. Contains model
        visibilities.
    u_true, v_true : array, unit=[k\lambda]
        Data u- and v-coordinates
    V_true : array, unit=[Jy]
        Data visibility amplitudes
    channel : int, default=0
        Channel (of `model`) to use to calculate residual visibilities

    Returns
    -------
    vis_resid : array of complex
        Model loose residual visibility amplitudes of the form
        Re(V) + 1j * Im(V)
    """
    nufft = NuFFT(coords=model.coords, nchan=model.nchan, uu=u_true, vv=v_true)

    vis_model = nufft(model.icube())
    # convert to numpy, select channel
    vis_model = vis_model.detach().numpy()[channel]

    vis_resid: NDArray[complexfloating[Any, Any]]
    vis_resid = V_true - vis_model

    return vis_resid
