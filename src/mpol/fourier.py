from __future__ import annotations


import numpy as np
import torch
import torch.fft  # to avoid conflicts with old torch.fft *function*
import torchkbnufft
from torch import nn

from mpol.exceptions import DimensionMismatchError

from mpol import utils
from mpol.coordinates import GridCoords


class FourierCube(nn.Module):
    r"""
    This layer performs the FFT of an ImageCube and stores the corresponding dense FFT
    output as a cube. If you are using this layer in a forward-modeling RML workflow,
    because the FFT of the model is essentially stored as a grid, you will need to make
    the loss function calculation using a gridded loss function (e.g.,
    :func:`mpol.losses.nll_gridded`) and a gridded dataset (e.g.,
    :class:`mpol.datasets.GriddedDataset`).

    Parameters
    ----------
    coords : :class:`~mpol.coordinates.GridCoords`
        object containing image dimensions
    persistent_vis : bool
        should the visibility cube be stored as part of
        the module  s `state_dict`? If `True`, the state of the UV grid will be
        stored. It is recommended to use `False` for most applications, since the
        visibility cube will rarely be a direct parameter of the model.

    """

    def __init__(self, coords: GridCoords, persistent_vis: bool = False):
        super().__init__()

        self.coords = coords

        self.register_buffer("vis", None, persistent=persistent_vis)
        self.vis: torch.Tensor

    def forward(self, packed_cube: torch.Tensor) -> torch.Tensor:
        """
        Perform the FFT of the image cube on each channel.

        Parameters
        ----------
        cube : :class:`torch.Tensor` of :class:`torch.double` of shape ``(nchan, npix, npix)``
            A 'packed' tensor. For example, an image cube from
            :meth:`mpol.images.ImageCube.forward`

        Returns
        -------
        :class:`torch.Tensor` of :class:`torch.double` of shape ``(nchan, npix, npix)``.
            The FFT of the image cube, in packed format.
        """

        # make sure the cube is 3D
        assert packed_cube.dim() == 3, "cube must be 3D"

        # the self.cell_size prefactor (in arcsec) is to obtain the correct output units
        # since it needs to correct for the spacing of the input grid.
        # See MPoL documentation and/or TMS Eqn A8.18 for more information.
        self.vis = self.coords.cell_size**2 * torch.fft.fftn(packed_cube, dim=(1, 2))

        return self.vis

    @property
    def ground_vis(self) -> torch.Tensor:
        r"""
        The visibility cube in ground format cube fftshifted for plotting with
        ``imshow``.

        Returns
        -------
        :class:`torch.Tensor` of :class:`torch.complex128` of shape ``(nchan, npix, npix)``
            complex-valued FFT of the image cube (i.e., the visibility cube), in
            'ground' format.
        """

        return utils.packed_cube_to_ground_cube(self.vis)

    @property
    def ground_amp(self) -> torch.Tensor:
        r"""
        The amplitude of the cube, arranged in unpacked format corresponding to the FFT
        of the sky_cube. Array dimensions for plotting given by ``self.coords.vis_ext``.

        Returns
        -------
        :class:`torch.Tensor` of :class:`torch.double` of shape ``(nchan, npix, npix)``
            amplitude cube in 'ground' format.
        """
        return torch.abs(self.ground_vis)

    @property
    def ground_phase(self) -> torch.Tensor:
        r"""
        The phase of the cube, arranged in unpacked format corresponding to the FFT of
        the sky_cube. Array dimensions for plotting given by ``self.coords.vis_ext``.

        Returns
        -------
        :class:`torch.Tensor` of :class:`torch.double` of shape ``(nchan, npix, npix)``
            phase cube in 'ground' format (:math:`[-\pi,\pi)`).
        """
        return torch.angle(self.ground_vis)


class NuFFT(nn.Module):
    r"""
    This layer translates input from an :class:`mpol.images.ImageCube` to loose,
    ungridded samples of the Fourier plane, corresponding to the :math:`u,v` locations
    provided. This layer is different than a :class:`mpol.Fourier.FourierCube` in that,
    rather than producing the dense cube-like output from an FFT routine, it utilizes
    the non-uniform FFT or 'NuFFT' to interpolate directly to discrete :math:`u,v`
    locations. This is implemented using the KbNufft routines of the `TorchKbNufft
    <https://torchkbnufft.readthedocs.io/en/stable/index.html>`_ package.

    Args:
        coords (GridCoords): an object already instantiated from the GridCoords class.
            If providing this, cannot provide ``cell_size`` or ``npix``.
        nchan (int): the number of channels in the :class:`mpol.images.ImageCube`.
            Default = 1.
    """

    def __init__(
        self,
        coords: GridCoords,
        nchan: int = 1,
    ):
        super().__init__()

        self.coords = coords
        self.nchan = nchan

        # initialize the non-uniform FFT object
        self.nufft_ob = torchkbnufft.KbNufft(
            im_size=(self.coords.npix, self.coords.npix)
        )

    def _lambda_to_radpix(self, lam: torch.Tensor) -> torch.Tensor:
        r"""Convert a spatial frequency in units of :math:`\lambda` to
        'radians/sky pixel,' using the pixel cell_size provided by ``self.coords.dl``.

        Args:
            lam (torch.Tensor): spatial frequency in units of :math:`\lambda`.

        Returns:
            torch.Tensor: spatial frequency measured in units of radian per sky pixel

        These concepts can be a little confusing because there are two angular measures
        at play.

        1. The first is the normal angular sky coordinate, normally measured in
        arcseconds for typical sources observed by ALMA or the VLA. Arcseconds, being
        an angular coordinate, can equivalently be expressed in units of radians. To
        avoid confusion, we will call this angular measurement 'sky radians.'
        Alternatively, for a given image grid, this same sky coordinate could be
        expressed in units of sky pixels.
        2. The second is the spatial frequency of some image-plane function,
        :math:`I_\nu(l,m)`, which we could quote in units of 'cycles per arcsecond' or
        'cycles per sky pixel,' for example. With a radio interferometer, spatial
        frequencies are typically quoted in units of the observing wavelength, i.e.,
        :math:`\lambda`. If the field of view of the image is small, thanks to
        the small-angle approximation, units of :math:`\lambda` are directly equivalent
        to 'cycles per sky radian.' The second angular measure comes about when
        converting the spatial frequency from a linear measure of frequency
        'cycles per sky radian' to an angular measure of frequency 'radians per sky
        radian' or 'radians per sky pixel.'

        The TorchKbNufft package expects k-trajectory vectors in units of 'radians per
        sky pixel.' This routine helps convert spatial frequencies from their default
        unit (:math:`\lambda`) into 'radians per sky pixel' using the pixel cell_size as
        provided by ``self.coords.dl``.
        """

        # lambda is equivalent to cycles per sky radian
        # convert from 'cycles per sky radian' to 'radians per sky radian'
        u_rad_per_rad = lam * 2 * np.pi  # [radians / sky radian]

        # size of pixel in radians
        # self.coords.dl  # [sky radians/pixel]

        # convert from 'radians per sky radian' to 'radians per sky pixel'
        # assumes pixels are square and dl and dm are interchangeable
        u_rad_per_pix = u_rad_per_rad * self.coords.dl  # [radians / pixel]

        return u_rad_per_pix

    def _assemble_ktraj(self, uu: torch.Tensor, vv: torch.Tensor) -> torch.Tensor:
        r"""
        This routine converts a series of :math:`u, v` coordinates into a k-trajectory
        vector for the torchkbnufft routines. The dimensionality of the k-trajectory
        vector will influence how TorchKbNufft will perform the operations.

        * If ``uu`` and ``vv`` have a 1D shape of (``nvis``), then it will be assumed
            that the spatial frequencies can be treated as constant with channel. This
            will result in a ``k_traj`` vector that has shape (``2, nvis``), such that
            parallelization will be across the image cube ``nchan`` dimension using the
            'coil' dimension of the TorchKbNufft package.
        * If the ``uu`` and ``vv`` have a 2D shape of (``nchan, nvis``), then it will
            be assumed that the spatial frequencies are different for each channel, and
            the spatial frequencies provided for each channel will be used. This will
            result in a ``k_traj`` vector that has shape (``nchan, 2, nvis``), such that
            parallelization will be across the image cube ``nchan`` dimension using the
            'batch' dimension of the TorchKbNufft package.

        Args:
            uu (1D or 2D torch.Tensor array): u (East-West) spatial frequency
                coordinate [:math:`\lambda`]
            vv (1D or 2D torch.Tensor array): v (North-South) spatial frequency
                coordinate [:math:`\lambda`]

        Returns:
            k_traj (torch.Tensor): a k-trajectory vector with shape
        """

        # if uu and vv are 1D dimension, then we assume 'same_uv'
        same_uv = (uu.ndim == 1) and (vv.ndim == 1)

        uu_radpix = self._lambda_to_radpix(uu)
        vv_radpix = self._lambda_to_radpix(vv)

        # torchkbnufft uses a [nbatch, ncoil, npix, npix] scheme
        # same_uv will parallelize across the coil dimension.
        if same_uv:
            # k-trajectory needs to be packed the way the image is packed (y,x), so
            # the trajectory needs to be packed (v, u)
            # if TorchKbNufft receives a k-traj tensor of shape (2, nvis),
            # it will parallelize across the coil dimension, assuming
            # that the k-traj is the same for all coils/channels.
            # interim convert to numpy array because of torch warning about speed
            # k_traj = torch.tensor(np.array([vv_radpix, uu_radpix]))
            k_traj = torch.vstack((vv_radpix, uu_radpix))
            return k_traj

        # !same_uv will parallelize across the batch dimension.
        # in this case, we are given two tensors of shape (nchan, nvis)
        # first, augment each tensor individually to create a (nbatch, 1, nvis) tensor,
        # where nbatch == nchan
        # then, concatenate the tensors along the axis=1 dimension.
        if uu_radpix.shape[0] != self.nchan:
            raise DimensionMismatchError(
                f"nchan of uu ({uu_radpix.shape[0]}) is more than one but different than that used to initialize the NuFFT layer ({self.nchan})"
            )

        if vv_radpix.shape[0] != self.nchan:
            raise DimensionMismatchError(
                f"nchan of vv ({vv_radpix.shape[0]}) is more than one but different than that used to initialize the NuFFT layer ({self.nchan})"
            )

        uu_radpix_aug = torch.unsqueeze(uu_radpix, 1)
        vv_radpix_aug = torch.unsqueeze(vv_radpix, 1)
        # if TorchKbNufft receives a k-traj tensor of shape (nbatch, 2, nvis), it will
        # parallelize across the batch dimension
        k_traj = torch.cat([vv_radpix_aug, uu_radpix_aug], dim=1)

        return k_traj

    def forward(
        self,
        packed_cube: torch.Tensor,
        uu: torch.Tensor,
        vv: torch.Tensor,
        sparse_matrices: bool = False,
    ) -> torch.Tensor:
        r"""
        Perform the FFT of the image cube for each channel and interpolate to the ``uu``
        and ``vv`` points. This call should automatically take the best
        parallelization option as indicated by the shape of the ``uu`` and ``vv``
        points. In general, you probably do not want to provide baselines that include
        Hermitian pairs.

        Parameters
        ----------
        packed_cube : :class:`torch.Tensor` of :class:`torch.double`
            shape ``(nchan, npix, npix)``). The cube
            should be a "prepacked" image cube, for example,
            from :meth:`mpol.images.ImageCube.forward`
        uu : :class:`torch.Tensor` of :class:`torch.double`
            2D array of the u (East-West) spatial frequency coordinate
            [:math:`\lambda`] of shape ``(nchan, npix)``
        vv : :class:`torch.Tensor` of :class:`torch.double`
            2D array of the v (North-South) spatial frequency coordinate
            [:math:`\lambda`] (must be the same shape as uu)
        sparse_matrices : bool
            If False, use the default table-based interpolation
            of TorchKbNufft.If True, use TorchKbNuFFT sparse matrices (generally
            slower but more accurate).  Note that sparse matrices are incompatible
            with multi-channel `uu` and `vv` arrays (see below).

        Returns
        -------
        :class:`torch.Tensor` of :class:`torch.complex128`
            Fourier samples of shape ``(nchan, nvis)``, evaluated at the ``uu``,
            ``vv`` points

        **Dimensionality**: You should consider the dimensionality of your image and
        your visibility samples when using this method. If your image has multiple
        channels (``nchan > 1``), there is the possibility that the :math:`u,v` sample
        locations corresponding to each channel may be different. In ALMA/VLA
        applications, this can arise when continuum observations are taken over
        significant bandwidth, since the spatial frequency sampled by any pair of
        antennas is wavelength-dependent

        .. math::

            u = \frac{D}{\lambda},

        where :math:`D` is the projected baseline (measured in, say, meters) and
        :math:`\lambda` is the observing wavelength. In this application, the
        image-plane model could be the same for each channel, or it may vary with
        channel (necessary if the spectral slope of the source is significant).

        On the other hand, with spectral line observations it will usually be the case
        that the total bandwidth of the observations is small enough such that the
        :math:`u,v` sample locations could be considered as the same for each channel.
        In spectral line applications, the image-plane model usually varies
        substantially with each channel.

        This routine will determine whether the spatial frequencies are treated as
        constant based upon the dimensionality of the ``uu`` and ``vv`` input arguments.

        * If ``uu`` and ``vv`` have a shape of (``nvis``), then it will be assumed that
            the spatial frequencies can be treated as constant with channel (and will
            invoke parallelization across the image cube ``nchan`` dimension using the
            'coil' dimension of the TorchKbNufft package).
        * If the ``uu`` and ``vv`` have a shape of (``nchan, nvis``), then it will be
            assumed that the spatial frequencies are different for each channel, and the
            spatial frequencies provided for each channel will be used (and will invoke
            parallelization across the image cube ``nchan`` dimension using the 'batch'
            dimension of the TorchKbNufft package).

        Note that there is no straightforward, computationally efficient way to proceed
        if there are a different number of spatial frequencies for each channel. The
        best approach is likely to construct ``uu`` and ``vv`` arrays that have a shape
        of (``nchan, nvis``), such that all channels are padded with bogus :math:`u,v`
        points to have the same length ``nvis``, and you create a boolean mask to keep
        track of which points are valid. Then, when this routine returns data points of
        shape (``nchan, nvis``), you can use that boolean mask to select only the valid
        :math:`u,v` points points.

        **Interpolation mode**: You may choose the type of interpolation mode that
        KbNufft uses under the hood by changing the boolean value of
        ``sparse_matrices``. If ``sparse_matrices=False``, this routine will use the
        default table-based interpolation of TorchKbNufft. If ``sparse_matrices=True``,
        the routine will calculate sparse matrices (which can be stored for later
        operations, as in {class}`~mpol.fourier.NuFFTCached`) and use them for the
        interpolation. This approach is likely to be more accurate but also slower. If
        Note that as of TorchKbNuFFT version 1.4.0, sparse matrices are not yet
        available when parallelizing using the 'batch' dimension --- this will result
        in a warning. For most applications, we anticipate the accuracy of the
        table-based interpolation to be sufficiently accurate, but this could change
        depending on your problem.
        """

        # make sure that the nchan assumptions for the ImageCube and the NuFFT
        # setup are the same
        if packed_cube.size(0) != self.nchan:
            raise DimensionMismatchError(
                f"nchan of ImageCube ({packed_cube.size(0)}) is different than that used to initialize NuFFT layer ({self.nchan})"
            )

        # "unpack" the cube, but leave it flipped
        # NuFFT routine expects a "normal" cube, not an fftshifted one
        shifted = torch.fft.fftshift(packed_cube, dim=(1, 2))

        # convert the cube to a complex type, since this is required by TorchKbNufft
        complexed = shifted.type(torch.complex128)

        k_traj = self._assemble_ktraj(uu, vv)

        # if uu and vv are 1D dimension, then we assume 'same_uv'
        same_uv = (uu.ndim == 1) and (vv.ndim == 1)

        # torchkbnufft uses a [nbatch, ncoil, npix, npix] scheme
        # same_uv will parallelize across the coil dimension.

        # Consider how the similarity of the spatial frequency samples should be
        # treated. We already took care of this on the k_traj side, since we set
        # the shapes. But this also needs to be taken care of on the image side.
        #   * If we plan to parallelize with the coil dimension, then we need an
        #     image with shape (1, nchan, npix, npix).
        #   * If we plan to parallelize using the batch dimension, then we need
        #     an image with shape (nchan, 1, npix, npix).

        if same_uv:
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

        # check that the user is following the documentation
        if not same_uv and sparse_matrices:
            import warnings

            warnings.warn(
                "Provided uu and vv arrays are multi-dimensional, suggesting an "
                "intent to parallelize using the 'batch' dimension. This feature "
                "is not yet available in TorchKbNuFFT v1.4.0 with sparse matrix "
                "interpolation (sparse_matrices=True), therefore we are proceeding "
                "with table interpolation (sparse_matrices=False). You may wish to "
                "fix your function call, because this warning slows down your code.",
                category=RuntimeWarning,
            )
            sparse_matrices = False

        # calculate the sparse_matrices, if requested
        if sparse_matrices:
            real_interp_mat, imag_interp_mat = torchkbnufft.calc_tensor_spmatrix(
                k_traj, im_size=(self.coords.npix, self.coords.npix)
            )

            output: torch.Tensor = self.coords.cell_size**2 * self.nufft_ob(
                expanded, k_traj, interp_mats=((real_interp_mat, imag_interp_mat))
            )

        else:
            output = self.coords.cell_size**2 * self.nufft_ob(expanded, k_traj)

        # squeezed shape: [nchan, npix, npix]
        output = torch.squeeze(output, dim=altered_dimension)

        return output


class NuFFTCached(NuFFT):
    r"""
    This layer is similar to the :class:`mpol.fourier.NuFFT`, but provides extra 
    functionality to cache the sparse matrices for a specific set of ``uu`` and ``vv``
    points specified at initialization. 
    
    For repeated evaluations of this layer (as might exist within an optimization loop),
    ``sparse_matrices=True`` is likely to be the more accurate and faster choice. If
    ``sparse_matrices=False``, this routine will use the default table-based
    interpolation of TorchKbNufft. Note that as of TorchKbNuFFT version 1.4.0, sparse
    matrices are not yet available when parallelizing using the 'batch' dimension ---
    this will result in a warning.

    Args:
        coords (GridCoords): an object already instantiated from the GridCoords class.
            If providing this, cannot provide ``cell_size`` or ``npix``.
        nchan (int): the number of channels in the :class:`mpol.images.ImageCube`.
            Default = 1.
        uu (array-like): a length ``nvis`` array (not including Hermitian pairs) of the
            u (East-West) spatial frequency coordinate [klambda]
        vv (array-like): a length ``nvis`` array (not including Hermitian pairs) of the
            v (North-South) spatial frequency coordinate [klambda]

    """

    def __init__(
        self,
        coords: GridCoords,
        uu: torch.Tensor,
        vv: torch.Tensor,
        nchan: int = 1,
        sparse_matrices: bool = True,
    ):
        super().__init__(coords, nchan)

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

        self.sparse_matrices = sparse_matrices
        self.same_uv = same_uv

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

    def forward(self, cube):
        r"""
        Perform the FFT of the image cube for each channel and interpolate to the
        ``uu`` and ``vv`` points set at layer initialization. This call should
        automatically take the best parallelization option as set by the shape of the
        ``uu`` and ``vv`` points.

        Args:
            cube (torch.double tensor): of shape ``(nchan, npix, npix)``). The cube
                should be a "prepacked" image cube, for example, from
                :meth:`mpol.images.ImageCube.forward`

        Returns:
            torch.complex tensor: of shape ``(nchan, nvis)``, Fourier samples evaluated
                corresponding to the ``uu``, ``vv`` points set at initialization.
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


def generate_fake_data(
    packed_cube: torch.Tensor,
    coords: GridCoords,
    uu: torch.Tensor,
    vv: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Create a fake dataset from a supplied packed tensor cube using
    :class:`mpol.fourier.NuFFT`. See :ref:`mock-dataset-label` for more details on how
    to prepare a generic image to a `packed_cube`.

    The ``uu`` and ``vv`` baselines can either be 1D or 2D, depending on the desired
    broadcasting behavior from the :class:`mpol.fourier.NuFFT`.

    If the ``weight`` array is 1D, the routine assumes the weights will be broadcasted
    to all ``nchan``. Otherwise, provide a 2D weight array.

    Parameters
    ----------
    packed_cube : :class:`torch.Tensor` of `class:`torch.double`
        the image in "packed" format with shape (`nchan`, `npix`, `npix`)
    coords : :class:`mpol.coordinates.GridCoords`
    uu : :class:`torch.Tensor` of `class:`torch.double`
        array of u spatial frequency coordinates,
        not including Hermitian pairs. Units of [:math:`\lambda`]
    vv : :class:`torch.Tensor` of `class:`torch.double`
        array of v spatial frequency coordinates,
        not including Hermitian pairs. Units of [:math:`\lambda`]
    weight : :class:`torch.Tensor` of `class:`torch.double`
        shape ``(nchan, nvis)`` array of thermal weights
        :math:`w_i = 1/\sigma_i^2`. Units of [:math:`1/\mathrm{Jy}^2`] Will broadcast
        from 1D to 2D if necessary.

    Returns
    -------
    :class:`torch.Tensor` of `class:`torch.complex128`
        A 2-tuple of the fake data. The first array is the mock dataset
        including noise, the second array is the mock dataset without added noise.
        Each array is shape ``(nchan, npix, npix)``.
    """

    # instantiate a NuFFT object based on the
    nchan, npix, _ = packed_cube.size()
    assert coords.npix == npix, "npix for packed_cube {:} and coords {:} differ".format(
        nchan, coords.npix
    )
    nufft = NuFFT(coords=coords, nchan=nchan)

    # carry it forward to the visibilities, which will be (nchan, nvis)
    vis_noiseless: torch.Tensor
    vis_noiseless = nufft(packed_cube, uu, vv)

    # generate complex noise
    # we could use torch.normal with complex quantities directly, but they treat
    # a complex normal with std of 1 as having an amplitude of 1.
    # wheras ALMA "weights" have the definition that their scatter is for the reals
    # (or imaginaries). So there is a factor of sqrt(2) between them.
    # we split things up, work with real quantities, and then combine later
    mean = torch.zeros(vis_noiseless.size())

    # broadcast weight to (nchan, nvis) if it isn't already
    weight = torch.broadcast_to(weight, vis_noiseless.size())
    sigma = torch.sqrt(1 / weight)

    noise_re = torch.normal(mean, sigma)
    noise_im = torch.normal(mean, sigma)
    noise = torch.complex(noise_re, noise_im)

    # add to data
    vis_noise = vis_noiseless + noise

    return vis_noise, vis_noiseless
