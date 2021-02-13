import numpy as np
from numpy.fft import ifft2, fftfreq, fftshift, ifftshift, rfftfreq
from .constants import arcsec
from .utils import get_max_spatial_freq, get_maximum_cell_size
from .coordinates import GridCoords, _setup_coords
from .datasets import GriddedDataset


class Gridder:
    r"""
    The Gridder object uses desired image dimensions (via the ``cell_size`` and ``npix`` arguments) to define a corresponding Fourier plane grid as a :class:`.GridCoords` object. A pre-computed :class:`.GridCoords` can be supplied in lieu of ``cell_size`` and ``npix``, but all three arguments should never be supplied at once. For more details on the properties of the grid that is created, see the :class:`.GridCoords` documentation.

    The :class:`.Gridder` object accepts "loose" *ungridded* visibility data and stores the arrays to the object as instance attributes. The input visibility data should be the set of visibilities over the full :math:`[-u,u]` and :math:`[-v,v]` domain, the Gridder will automatically augment the dataset to include the complex conjugates. 

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

        nchan = len(uu)
        _setup_coords(self, cell_size, npix, coords, nchan)

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

        # expand the vectors to include complex conjugates
        uu_full = np.concatenate([uu, -uu], axis=1)
        vv_full = np.concatenate([vv, -vv], axis=1)

        # make sure we still fit into the grid
        self.coords.check_data_fit(uu_full, vv_full)

        self.uu = uu_full
        self.vv = vv_full
        self.weight = np.concatenate([weight, weight], axis=1)
        self.data_re = np.concatenate([data_re, data_re], axis=1)
        self.data_im = np.concatenate([data_im, -data_im], axis=1)

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
            weight (iterable): ``(nchan, nvis)`` list of 1D arrays of weights of shape to use in the histogramming.

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
        Grid the loose data visibilities to the Fourier grid in preparation for imaging.

        Args:
            weighting (string): The type of cell averaging to perform. Choices of ``"natural"``, ``"uniform"``, or ``"briggs"``, following CASA tclean. If ``"briggs"``, also specify a robust value.
            robust (float): If ``weighting='briggs'``, specify a robust value in the range [-2, 2]. ``robust=-2`` approxmately corresponds to uniform weighting and ``robust=2`` approximately corresponds to natural weighting. 
            taper_function (function reference): a function assumed to be of the form :math:`f(u,v)` which calculates a prefactor in the range :math:`[0,1]` and premultiplies the visibility data. The function must assume that :math:`u` and :math:`v` will be supplied in units of :math:`\mathrm{k}\lambda`. By default no taper is applied.
        """

        if taper_function is None:
            tapering_weight = np.ones_like(self.weight)
        else:
            tapering_weight = taper_function(self.uu, self.vv)

        # create the cells as edges around the existing points
        # note that at this stage, the UV grid is strictly increasing
        # when in fact, later on, we'll need to fftshift for the FFT
        cell_weight = self._histogram_cube(self.weight)

        # boolean index for cells that *contain* visibilities
        self.mask = cell_weight > 0.0

        # calculate the density weights
        # the density weights have the same shape as the re, im samples.
        if weighting == "natural":
            density_weight = np.ones_like(self.weight)
        elif weighting == "uniform":
            # cell_weight is (nchan, ncell_v, ncell_u)
            # self.index_v, self.index_u are (nchan, nvis)
            # we want density_weights to be (nchan, nvis)
            density_weight = 1 / np.array(
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
                np.sum(cell_weight ** 2, axis=(1, 2)) / np.sum(self.weight, axis=1)
            )

            # the robust weight corresponding to the cell
            cell_robust_weight = 1 / (1 + cell_weight * f_sq[:, np.newaxis, np.newaxis])

            # zero out cells that have no visibilities
            # to prevent normalization error in next step
            cell_robust_weight[~self.mask] = 0

            # now assign the cell robust weight to each visibility within that cell
            density_weight = np.array(
                [
                    cell_robust_weight[i][self.index_v[i], self.index_u[i]]
                    for i in range(self.nchan)
                ]
            )
        else:
            raise ValueError(
                "weighting must be specified as one of 'natural', 'uniform', or 'briggs'"
            )

        # the factor of 2 in the denominator is needed because
        # we are approximating the Eqn 3.8 of Briggs' thesis
        # we need to sum over the Hermitian quantities in the
        # normalization constant.
        self.C = 1 / np.sum(tapering_weight * density_weight * self.weight, axis=1)

        # grid the reals and imaginaries separately
        self.data_re_gridded = self._histogram_cube(
            self.data_re * tapering_weight * density_weight * self.weight
        )

        self.data_im_gridded = self._histogram_cube(
            self.data_im * tapering_weight * density_weight * self.weight
        )

        self.vis_gridded = self.data_re_gridded + self.data_im_gridded * 1.0j

        # the beam is the response to a point source, which is data_re = constant, data_im = 0
        # so we save time and only calculate the reals, because gridded_beam_im = 0
        self.re_gridded_beam = self._histogram_cube(
            tapering_weight * density_weight * self.weight
        )

        # instantiate uncertainties for each averaged visibility.
        if weighting == "uniform" and robust == None and taper_function is None:
            self.weight_gridded = cell_weight
        else:
            self.weight_gridded = None

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

        beam = self._fliplr_cube(
            np.fft.fftshift(
                self.coords.npix ** 2
                * np.fft.ifft2(
                    np.fft.fftshift(
                        self.C[:, np.newaxis, np.newaxis] * self.re_gridded_beam,
                        axes=(1, 2),
                    )
                ),
                axes=(1, 2),
            )
        )

        assert (
            np.max(beam.imag) < 1e-10
        ), "Dirty beam contained substantial imaginary values, check input visibilities, otherwise raise a github issue."

        self.beam = beam.real

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

        img = self._fliplr_cube(
            np.fft.fftshift(
                self.coords.npix ** 2
                * np.fft.ifft2(
                    np.fft.fftshift(
                        self.C[:, np.newaxis, np.newaxis] * self.vis_gridded,
                        axes=(1, 2),
                    )
                ),
                axes=(1, 2),
            )
        )  # Jy/beam

        assert (
            np.max(img.imag) < 1e-10
        ), "Dirty image contained substantial imaginary values, check input visibilities, otherwise raise a github issue."

        self.img = img.real

        return self.img

    def to_pytorch_dataset(self):
        """
        Export gridded visibilities to a PyTorch dataset object.

        Returns:
            :class:`~mpol.datasets.GriddedDataset` with gridded visibilities.
        """

        assert (
            self.weight_gridded is not None
        ), "To export with uncertainties, first grid visibilities with weighting='uniform', no tapering function, and robust=None. Otherwise, data weights are not defined."

        return GriddedDataset(
            coords=self.coords,
            nchan=self.nchan,
            vis_gridded=self.vis_gridded,
            weight_gridded=self.weight_gridded,
            mask=self.mask,
        )

