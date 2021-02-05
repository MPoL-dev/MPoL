import numpy as np
from numpy.fft import ifft2, fftfreq, fftshift, ifftshift, rfftfreq
from .constants import arcsec
from .utils import get_max_spatial_freq, get_maximum_cell_size


class GridCoords:
    r"""
    The GridCoords object uses desired image dimensions (via the ``cell_size`` and ``npix`` arguments) to define a corresponding Fourier plane grid. 
    
    Args:
        cell_size (float): width of a single square pixel in [arcsec]
        npix (int): number of pixels in the width of the image

    For real images, the Fourier grid is minimally defined by an RFFT grid over the domain :math:`[0,+u]`, :math:`[-v,+v]`. 

    Images (and their corresponding Fourier transform quantities) are represented as two-dimensional arrays packed as ``[y, x]`` and ``[v, u]``.  This means that an image with dimensions ``(npix, npix)`` will have a corresponding RFFT Fourier grid with shape ``(npix, npix//2 + 1)`` because the RFFT is performed over the trailing coordinate, in this case :math:`u`. The native :class:`~mpol.gridding.GridCoords` representation assumes the Fourier grid (and thus image) are laid out as one might normally expect an image (i.e., no ``np.fft.fftshift`` has been applied).

    After the object is initialized, instance variables can be accessed, for example
    
    >>> myCoords = GridCoords(cell_size=0.005, 512)
    >>> myCoords.img_ext
    
    :ivar dl: image-plane cell spacing in RA direction (assumed to be positive) [radians]
    :ivar dm: image-plane cell spacing in DEC direction [radians]
    :ivar img_ext: The length-4 list of (left, right, bottom, top) expected by routines like ``matplotlib.pyplot.imshow`` in the ``extent`` parameter assuming ``origin='lower'``. Units of [arcsec]
    :ivar du: Fourier-plane cell spacing in East-West direction [:math:`\mathrm{k}\lambda`]
    :ivar dv: Fourier-plane cell spacing in North-South direction [:math:`\mathrm{k}\lambda`]
    :ivar u_centers: 1D array of cell centers in East-West direction [:math:`\mathrm{k}\lambda`]. 
    :ivar v_centers: 1D array of cell centers in North-West direction [:math:`\mathrm{k}\lambda`]. 
    :ivar u_edges: 1D array of cell edges in East-West direction [:math:`\mathrm{k}\lambda`]. 
    :ivar v_edges: 1D array of cell edges in North-South direction [:math:`\mathrm{k}\lambda`]. 
    :ivar u_bin_min: minimum u edge [:math:`\mathrm{k}\lambda`]
    :ivar u_bin_max: maximum u edge [:math:`\mathrm{k}\lambda`]
    :ivar v_bin_min: minimum v edge [:math:`\mathrm{k}\lambda`]
    :ivar v_bin_max: maximum v edge [:math:`\mathrm{k}\lambda`]
    :ivar max_grid: maximum spatial frequency enclosed by Fourier grid [:math:`\mathrm{k}\lambda`]
    :ivar vis_ext: length-4 list of (left, right, bottom, top) expected by routines like ``matplotlib.pyplot.imshow`` in the ``extent`` parameter assuming ``origin='lower'``. Units of [:math:`\mathrm{k}\lambda`]
    """

    def __init__(self, cell_size, npix):
        # set up the bin edges, centers, etc.
        assert npix % 2 == 0, "Image must have an even number of pixels"
        assert cell_size > 0, "cell_size must be positive"

        self.cell_size = cell_size  # arcsec
        self.npix = npix
        self.ncell_u = self.npix // 2 + 1
        self.ncell_v = self.npix

        # calculate the image extent
        # say we had 10 pixels representing centers -5, -4, -3, ...
        # it should go from -5.5 to +4.5
        lmax = cell_size * (self.npix // 2 - 0.5)
        lmin = -cell_size * (self.npix // 2 + 0.5)
        self.img_ext = [lmax, lmin, lmin, lmax]  # arcsecs

        self.dl = cell_size * arcsec  # [radians]
        self.dm = cell_size * arcsec  # [radians]

        # the output spatial frequencies of the FFT routine
        self.du = 1 / (self.npix * self.dl) * 1e-3  # [kλ]
        self.dv = 1 / (self.npix * self.dm) * 1e-3  # [kλ]

        # define the max/min of the RFFT grid
        # https://numpy.org/doc/stable/reference/generated/numpy.fft.rfftn.html#numpy.fft.rfftn
        # the real transform is performed over the last axis.
        # because we store images as [y, x]
        # this means we store visibilities as [v, u]
        # that means that the u dimension gets the real transform
        # and the v dimension gets the full transform
        int_u_edges = np.arange(self.ncell_u + 1) - 0.5
        int_v_edges = np.arange(self.ncell_v + 1) - self.ncell_v // 2 - 0.5

        # print("int_u_edges", int_u_edges)
        # print("int_v_edges", int_v_edges)
        self.u_edges = self.du * int_u_edges  # [kλ]
        self.v_edges = self.dv * int_v_edges  # [kλ]

        int_u_centers = np.arange(self.ncell_u)
        int_v_centers = np.arange(self.ncell_v) - self.npix // 2
        # print("int_u_centers", int_u_centers)
        # print("int_v_centers", int_v_centers)
        self.u_centers = self.du * int_u_centers  # [kλ]
        self.v_centers = self.dv * int_v_centers  # [kλ]

        self.v_bin_min = np.min(self.v_edges)
        self.v_bin_max = np.max(self.v_edges)

        self.u_bin_min = np.min(self.u_edges)
        self.u_bin_max = np.max(self.u_edges)

        self.vis_ext = [
            self.u_bin_min,
            self.u_bin_max,
            self.v_bin_min,
            self.v_bin_max,
        ]  # [kλ]

        # max freq supported by current grid
        self.max_grid = get_max_spatial_freq(self.cell_size, self.npix)

    def check_data_fit(self, uu, vv):
        r"""
        Test whether loose data visibilities fit within the Fourier grid defined by cell_size and npix.

        Args:
            uu (np.array): array of u spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
            vv (np.array): array of v spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]

        Returns: ``True`` if all visibilities fit within the Fourier grid defined by ``[u_bin_min, u_bin_max, v_bin_min, v_bin_max]``. Otherwise an ``AssertionError`` is raised on the first violated boundary.
        """

        # max freq in dataset
        max_uu_vv = np.max(np.abs(np.concatenate([uu, vv])))

        # max freq needed to support dataset
        max_cell_size = get_maximum_cell_size(max_uu_vv)

        assert (
            np.max(np.abs(uu)) < self.max_grid
        ), "Dataset contains uu spatial frequency measurements larger than those in the proposed model image. Decrease cell_size below {:} arcsec.".format(
            max_cell_size
        )
        assert (
            np.max(np.abs(vv)) < self.max_grid
        ), "Dataset contains vv spatial frequency measurements larger than those in the proposed model image. Decrease cell_size below {:} arcsec.".format(
            max_cell_size
        )

        return True


# wrappers to do the histogramming


class Gridder:
    r"""
    The Gridder object uses desired image dimensions (via the ``cell_size`` and ``npix`` arguments) to define a corresponding Fourier plane grid as a :class:`.GridCoords` object. A pre-computed :class:`.GridCoords` can be supplied in lieu of ``cell_size`` and ``npix``, but all three arguments should never be supplied at once. For more details on the properties of the grid that is created, see the :class:`.GridCoords` documentation.

    The :class:`.Gridder` object accepts "loose" *ungridded* visibility data and stores the arrays to the object as instance attributes. The input visibility data should be the set of visibilities over the full :math:`[-u,u]` and :math:`[-v,v]` domain, the Gridder will automatically augment the dataset to include the complex conjugates and shift all loose data into the RFFT domain (:math:`[0,u]` and :math:`[-v,v]`).

    Once the loose visibilities are attached, the user can decide how to 'grid', or average, them to a more compact representation on the Fourier grid using the :func:`~mpol.gridding.Gridder.grid_visibilities` routine.

    If your goal is to use these gridded visibilities in Regularized Maximum Likelihood imaging, you can export them to the appropriate PyTorch object using the :func:`~mpol.gridding.Gridder.to_pytorch_dataset` routine.

    If you just want to take a quick look at the rough image plane representation of the visibilities, you can view the 'dirty image' and the point spread function or 'dirty beam'. After the visibilities have been gridded with :func:`~mpol.gridding.Gridder.grid_visibilities`, these are available via the :func:`~mpol.gridding.Gridder.get_dirty_image` and :func:`~mpol.gridding.Gridder.get_dirty_beam` methods.

    Like the :class:`~mpol.images.ImageCube` class, the Gridder assumes that you are operating with a multi-channel set of visibilities. These routines will still work with single-channel 'continuum' visibilities, they will just have nchan = 1 in the first dimension of most products.

    Args:
        cell_size (float): width of a single square pixel in [arcsec]
        npix (int): number of pixels in the width of the image
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        uu (2d numpy array): (nchan, nvis) length array of u spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
        vv (2d numpy array): (nchan, nvis) length array of v spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
        weight (2d numpy array): (nchan, nvis) length array of thermal weights. Units of [:math:`1/\mathrm{Jy}^2`]
        data_re (2d numpy array): (nchan, nvis) length array of the real part of the visibility measurements. Units of [:math:`\mathrm{Jy}`]
        data_im (2d numpy array): (nchan, nvis) length array of the imaginary part of the visibility measurements. Units of [:math:`\mathrm{Jy}`]

    """

    def __init__(
        self,
        cell_size=None,
        npix=None,
        coords=None,
        uu=None,
        vv=None,
        weight=None,
        data_re=None,
        data_im=None,
    ):

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

        assert (
            uu.ndim == 2
        ), "Input data vectors should be 2D numpy arrays. If you have a continuum observation, make all data vectors 2D with shape (1, nvis)."
        shape = uu.shape

        for a in [vv, weight, data_re, data_im]:
            assert a.shape == shape, "All dataset inputs must be the same 2D shape."

        assert np.all(
            weight > 0.0
        ), "Not all thermal weights are positive, check inputs."

        assert data_re.dtype == np.float64, "data_re should be type np.float64"
        assert data_im.dtype == np.float64, "data_im should be type np.float64"

        # expand and overwrite the vectors to include complex conjugates
        uu_full = np.concatenate([uu, -uu], axis=1)
        vv_full = np.concatenate([vv, -vv], axis=1)
        weight_full = np.concatenate([weight, weight], axis=1)
        data_re_full = np.concatenate([data_re, data_re], axis=1)
        data_im_full = np.concatenate([data_im, -data_im], axis=1)

        # remove all visibilities that have uu < u_bin_bin

        # The RFFT outputs u in the range [0, +] and v in the range [-, +],
        # but the dataset contains measurements at u [-,+] and v [-, +].
        # Find all the u < 0 points and convert them via complex conj
        ind_u_neg = uu < 0.0
        uu[ind_u_neg] *= -1.0  # swap axes so all u > 0
        vv[ind_u_neg] *= -1.0  # swap axes
        data_im[ind_u_neg] *= -1.0  # complex conjugate
        self.coords.check_data_fit(uu=uu, vv=vv)

        # No need to duplicate and complex-conjugate visibilities
        # like with the full FFT. The RFFT naturally assumes that the complex
        # conjugates exist.

        # if all checks out, store these to the object
        self.uu = uu_full
        self.vv = vv_full
        self.weight = weight_full
        self.data_re = data_re_full
        self.data_im = data_im_full

        self.nchan = len(self.uu)

        # figure out which visibility cell each datapoint lands in, so that
        # we can later assign it the appropriate robust weight for that cell
        # do this by calculating the nearest cell index [0, N] for all samples
        self.index_u = np.array(
            [np.digitize(u_chan, self.coords.u_edges) - 1 for u_chan in self.uu]
        )
        self.index_v = np.array(
            [np.digitize(v_chan, self.coords.v_edges) - 1 for v_chan in self.vv]
        )

    def _histogram_channel(self, uu, vv, channel_weight):
        r"""
        Perform a 2D histogram over the Fourier grid defined by ``coords``.

        Args:
            uu (np.array): 1D array of East-West spatial frequency coordinates for a specific channel. Units of [:math:`\mathrm{k}\lambda`]
            vv: 1D array of North-South spatial frequency coordinates for a specific channel. Units of [:math:`\mathrm{k}\lambda`]
            channel_weight (np.array): 1D array of weights to use in the histogramming.

        """
        # order is swapped because of the way the image looks

        result = np.histogram2d(
            vv,
            uu,
            bins=[self.coords.v_edges, self.coords.u_edges],
            weights=channel_weight,
        )

        # only return the "H" value
        return result[0]

    def _histogram_cube(self, weight):
        r"""
        Perform a 2D histogram over the  Fourier grid defined by ``coords``, for all channels..

        Args:
            weight (np.array): 2D array of weights of shape ``(nchan, nvis)`` to use in the histogramming.

        """
        # calculate the histogrammed result for all channels
        cube = np.empty(
            (self.nchan, self.coords.ncell_v, self.coords.ncell_u), dtype=np.float64,
        )

        for i in range(self.nchan):
            cube[i] = self._histogram_channel(self.uu[i], self.vv[i], weight[i])

        return cube

    def grid_visibilities(self, weighting="uniform", robust=None, taper_function=None):
        r"""
        Grid the loose data visibilities to the Fourier grid.

        Args:
            weighting (string): The type of cell averaging to perform. Choices of ``"natural"``, ``"uniform"``, or ``"briggs"``, following CASA tclean. If ``"briggs"``, also specify a robust value.
            robust (float): If ``weighting='briggs'``, specify a robust value in the range [-2, 2].
            taper_function (function reference): a function assumed to be of the form :math:`f(u,v)` which calculates a prefactor in the range :math:`[0,1]` and premultiplies the visibility data. The function must assume that :math:`u` and :math:`v` will be supplied in units of :math:`\mathrm{k}\lambda`. By default no taper is applied.
        """

        if taper_function is None:
            tapering_weight = np.ones_like(self.weight)
        else:
            tapering_weight = taper_function(self.uu, self.vv)

        # create the cells as edges around the existing points
        # note that at this stage, the UV grid is strictly increasing
        # when in fact, later on, we'll need to fftshift the v axis
        # for the RFFT
        cell_weight = self._histogram_cube(self.weight)

        # calculate the density weights
        # the density weights have the same shape as the re, im samples.
        if weighting == "natural":
            density_weights = np.ones_like(self.weight)
        elif weighting == "uniform":
            # cell_weight is (nchan, ncell_v, ncell_u)
            # self.index_v, self.index_u are (nchan, nvis)
            # we want density_weights to be (nchan, nvis)
            density_weights = 1 / np.array(
                [
                    cell_weight[i][self.index_v[i], self.index_u[i]]
                    for i in range(self.nchan)
                ]
            )

        elif weighting == "briggs":
            if robust is None:
                raise ValueError(
                    "If 'briggs' weighting, a robust value must be specified between [-2, 2]."
                )
            assert (robust >= -2) and (
                robust <= 2
            ), "robust parameter must be in the range [-2, 2]"

            # implement robust weighting using the definition used in CASA
            # https://casa.nrao.edu/casadocs-devel/stable/imaging/synthesis-imaging/data-weighting

            # calculate the robust parameter f^2 for each channel
            f_sq = ((5 * 10 ** (-robust)) ** 2) / (
                np.sum(cell_weight ** 2, axis=(1, 2)) / np.sum(self.weight, axis=(1, 2))
            )

            # the robust weight corresponding to the cell
            cell_robust_weight = 1 / (1 + cell_weight * f_sq)

            # zero out cells that have no visibilities
            cell_robust_weight[cell_weight <= 0.0] = 0

            # now assign the cell robust weight to each visibility within that cell
            density_weights = cell_robust_weight[self.index_v, self.index_u]
        else:
            raise ValueError(
                "weighting must be specified as one of 'natural', 'uniform', or 'briggs'"
            )

        # the factor of 2 in the denominator is needed because
        # we are approximating the Eqn 3.8 of Briggs' thesis
        # we need to sum over the Hermitian quantities in the
        # normalization constant.
        self.debug_info = (tapering_weight[4], density_weights[4], self.weight[4])
        self.C = 1 / (
            2 * np.sum(tapering_weight * density_weights * self.weight, axis=1)
        )

        # grid the reals and imaginaries separately
        self.gridded_re = self._histogram_cube(
            self.data_re * tapering_weight * density_weights * self.weight
        )
        self.gridded_im = self._histogram_cube(
            self.data_im * tapering_weight * density_weights * self.weight
        )

        self.gridded_vis = self.gridded_re + self.gridded_im * 1.0j

        # the beam is the response to a point source, which is data_re = constant, data_im = 0
        # so we save time and only calculate the reals, because gridded_beam_im = 0
        self.gridded_beam_re = self._histogram_cube(
            tapering_weight * density_weights * self.weight
        )

        # instantiate uncertainties for each averaged visibility.
        # self.VV_uncertainty = values
        # self.VV_uncertainty = None, for routines which it's not implemented yet.

    def _fliplr_cube(self, cube):
        return cube[:, :, ::-1]

    def get_dirty_beam(self):
        """
        Compute the dirty beam corresponding to the gridded visibilities.

        Returns: numpy image cube with a dirty beam (PSF) for each channel. The units are in Jy/{dirty beam}, i.e., the peak is normalized to 1.0.
        """
        # if we're sticking to the dirty beam and image equations in Briggs' Ph.D. thesis,
        # no correction for du or dv prefactors needed here
        # that is because we are using the FFT to compute an already discretized equation, not
        # approximating a continuous equation.

        self.beam = self._fliplr_cube(
            np.fft.fftshift(
                self.coords.npix ** 2
                * np.fft.irfft2(
                    np.fft.fftshift(
                        self.C[:, np.newaxis, np.newaxis] * self.gridded_beam_re, axes=1
                    )
                ),
                axes=(1, 2),
            )
        )

        return self.beam

    def get_dirty_image(self, unit="Jy/beam"):
        """
        Calculate the dirty image.

        Args:
            unit (string): what unit should the image be in. Default is ``"Jy/beam"``. If ``"Jy/arcsec^2"``, then the effective area of the dirty beam will be used to convert from ``"Jy/beam"`` to ``"Jy/arcsec^2"``.

        Returns: (nchan, npix, npix) numpy array of the dirty image cube.
        """

        if unit not in ["Jy/beam", "Jy/arcsec^2"]:
            raise ValueError("Unknown unit", unit)

        self.img = self._fliplr_cube(
            np.fft.fftshift(
                self.coords.npix ** 2
                * np.fft.irfft2(
                    np.fft.fftshift(
                        self.C[:, np.newaxis, np.newaxis] * self.gridded_vis, axes=1
                    )
                ),
                axes=(1, 2),
            )
        )  # Jy/beam

        return self.img

    def to_pytorch_dataset(self):
        """
        Export gridded visibilities to a PyTorch dataset object
        """
        raise NotImplementedError()


class ChannelImager:
    def __init__(self, cell_size, npix, uu, vv, weights, re, im):
        """
        cell_size in arcsec
        uu, vv in klambda
        re, im in Jy
        """
        # set up the bin edges, centers, etc.
        assert npix % 2 == 0, "Image must have an even number of pixels"

        self.cell_size = cell_size
        self.npix = npix

        # calculate the image extent
        # say we had 10 pixels
        # it should go from -5.5 to +4.5
        lmax = cell_size * (self.npix // 2 - 0.5)
        lmin = -cell_size * (self.npix // 2 + 0.5)
        self.img_ext = [lmax, lmin, lmin, lmax]  # arcsecs

        self.dl = cell_size * arcsec  # [radians]
        self.dm = cell_size * arcsec  # [radians]

        # the output spatial frequencies of the FFT routine
        self.du = 1 / (self.npix * self.dl) * 1e-3  # klambda
        self.dv = 1 / (self.npix * self.dm) * 1e-3  # klambda

        int_edges = np.arange(self.npix + 1) - self.npix // 2 - 0.5
        self.u_edges = self.du * int_edges  # klambda
        self.v_edges = self.dv * int_edges

        int_centers = np.arange(self.npix) - self.npix // 2
        self.u_centers = self.du * int_centers
        self.v_centers = self.dv * int_centers

        v_bin_min = np.min(self.v_edges)
        v_bin_max = np.max(self.v_edges)
        u_bin_min = np.min(self.u_edges)
        u_bin_max = np.max(self.u_edges)

        self.vis_ext = [u_bin_min, u_bin_max, v_bin_min, v_bin_max]  # klambda

        assert uu.ndim == 1, "Input arrays must be 1-dimensional"

        assert np.all(
            np.array([len(arr) for arr in [vv, weights]]) == len(uu)
        ), "Input arrays are not the same length."

        # expand and overwrite the vectors to include complex conjugates
        self.uu = np.concatenate([uu, -uu])
        self.vv = np.concatenate([vv, -vv])
        self.weights = np.concatenate([weights, weights])
        self.re = np.concatenate([re, re])
        self.im = np.concatenate([im, -im])  # the complex conjugates

        # figure out which cell each visibility lands in, so that
        # we can assign it the appropriate robust weight for that cell
        # do this by calculating the nearest cell index [0, N] for all samples
        self.index_u = np.digitize(self.uu, self.u_edges) - 1
        self.index_v = np.digitize(self.vv, self.v_edges) - 1

    def grid_visibilities(self, weighting, robust=None, taper_function=None):
        """
        Specify weighting of "natural", "uniform", or "briggs", following CASA tclean. If briggs, specify robust in [-2, 2].

        If specifying a taper function, then it is assumed to be f(u, v).
        """

        if taper_function is None:
            tapering_weights = np.ones_like(self.weights)
        else:
            tapering_weights = taper_function(self.uu, self.vv)

        # create the cells as edges around the existing points
        # note that at this stage, the bins are strictly increasing
        # when in fact, later on, we'll need to put this into fftshift format for the FFT
        cell_weight, junk, junk = np.histogram2d(
            self.vv, self.uu, bins=[self.v_edges, self.u_edges], weights=self.weights,
        )

        # calculate the density weights
        # the density weights have the same shape as the re, im samples.
        if weighting == "natural":
            density_weights = np.ones_like(self.weights)
        elif weighting == "uniform":
            density_weights = 1 / cell_weight[self.index_v, self.index_u]
        elif weighting == "briggs":
            if robust is None:
                raise ValueError(
                    "If 'briggs' weighting, a robust value must be specified between [-2, 2]."
                )
            assert (robust >= -2) and (
                robust <= 2
            ), "robust parameter must be in the range [-2, 2]"

            # implement robust weighting using the definition used in CASA
            # https://casa.nrao.edu/casadocs-devel/stable/imaging/synthesis-imaging/data-weighting

            # calculate the robust parameter f^2
            f_sq = ((5 * 10 ** (-robust)) ** 2) / (
                np.sum(cell_weight ** 2) / np.sum(self.weights)
            )

            # the robust weight corresponding to the cell
            cell_robust_weight = 1 / (1 + cell_weight * f_sq)

            # zero out cells that have no visibilities
            cell_robust_weight[cell_weight <= 0.0] = 0

            # now assign the cell robust weight to each visibility within that cell
            density_weights = cell_robust_weight[self.index_v, self.index_u]
        else:
            raise ValueError(
                "weighting must be specified as one of 'natural', 'uniform', or 'briggs'"
            )

        self.debug_info = (tapering_weights, density_weights, self.weights)
        self.C = 1 / np.sum(tapering_weights * density_weights * self.weights)

        # grid the reals and imaginaries
        VV_g_real, junk, junk = np.histogram2d(
            self.vv,
            self.uu,
            bins=[self.v_edges, self.u_edges],
            weights=self.re * tapering_weights * density_weights * self.weights,
        )

        VV_g_imag, junk, junk = np.histogram2d(
            self.vv,
            self.uu,
            bins=[self.v_edges, self.u_edges],
            weights=self.im * tapering_weights * density_weights * self.weights,
        )

        self.VV_g = VV_g_real + VV_g_imag * 1.0j

        # do the beam too
        beam_V_real, junk, junk = np.histogram2d(
            self.vv,
            self.uu,
            bins=[self.v_edges, self.u_edges],
            weights=tapering_weights * density_weights * self.weights,
        )
        self.beam_V = beam_V_real

    def image_gridded_visibilities(self):

        # if we're sticking to Briggs' equations, no correction for du or dv prefactors needed here
        # that is because we are using the FFT to compute an already discretized equation, not
        # approximating a continuous equation.

        self.beam = np.fliplr(
            np.fft.fftshift(
                self.npix ** 2
                * np.fft.ifftn(np.fft.fftshift(self.C * self.beam_V), axes=(0, 1))
            )
        ).real  # Jy/radians^2

        self.img = np.fliplr(
            np.fft.fftshift(
                self.npix ** 2
                * np.fft.ifftn(np.fft.fftshift(self.C * self.VV_g), axes=(0, 1))
            )
        ).real  # Jy/radians^2
