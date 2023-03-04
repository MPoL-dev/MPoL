r"""The ``gridding`` module provides two core classes in :class:`mpol.gridding.DataAverager` and :class:`mpol.gridding.DirtyImager`."""


from __future__ import annotations

import warnings

import numpy as np
from fast_histogram import histogram as fast_hist

from mpol.coordinates import GridCoords
from mpol.exceptions import DataError, ThresholdExceededError, WrongDimensionError

from .datasets import GriddedDataset


def _check_data_inputs_2d(uu=None, vv=None, weight=None, data_re=None, data_im=None):
    """
    Check that all data inputs are the same shape, the weights are positive, and the data_re and data_im are floats.

    Make a reasonable effort to ensure that Hermitian pairs are *not* included.

    If the user supplied 1d vectors of shape ``(nvis,)``, make them all 2d with one channel, ``(1,nvis)``.

    """

    if not 1 <= uu.ndim <= 2:
        raise WrongDimensionError(
            "Input data vectors should be either 1D or 2D numpy arrays."
        )

    if not all(array.shape == uu.shape for array in [vv, weight, data_re, data_im]):
        raise WrongDimensionError(
            "All dataset inputs must be the same input shape and size."
        )

    if np.any(weight <= 0.0):
        raise ValueError("Not all thermal weights are positive, check inputs.")

    if data_re.dtype not in (np.single, np.double):
        raise TypeError("data_re should be of type np.single or np.double.")

    if data_im.dtype not in (np.single, np.double):
        raise TypeError("data_im should be of type np.single or np.double.")

    if uu.ndim == 1:
        uu = np.atleast_2d(uu)
        vv = np.atleast_2d(vv)
        weight = np.atleast_2d(weight)
        data_re = np.atleast_2d(data_re)
        data_im = np.atleast_2d(data_im)

    # check to see that uu, vv and data do not contain Hermitian pairs
    verify_no_hermitian_pairs(uu, vv, data_re + 1.0j * data_im)

    return uu, vv, weight, data_re, data_im


def verify_no_hermitian_pairs(uu, vv, data, test_vis=5, test_channel=0):
    r"""
    Check that the dataset does not contain Hermitian pairs. Because the sky brightness :math:`I_\nu` is real, the visibility function :math:`\mathcal{V}` is Hermitian, meaning that

    .. math::

        \mathcal{V}(u, v) = \mathcal{V}^*(-u, -v).

    Most datasets (e.g., those extracted from CASA) will only record one visibility measurement per baseline and not include the duplicate Hermitian pair (to save storage space). This routine attempts to determine if the dataset contains Hermitian pairs or not by choosing one data point at a time and then searching the dataset to see if its Hermitian pair exists. The routine will declare that a dataset contains Hermitian pairs or not after it has searched ``test_vis`` number of data points. If 0 Hermitian pairs have been found for all ``test_vis`` points, then the dataset will be declared to have no Hermitian pairs. If ``test_vis`` Hermitian pairs were found for ``test_vis`` points searched, then the dataset will be declared to have Hermitian pairs. If more than 0 but fewer than ``test_vis`` Hermitian pairs were found for ``test_vis`` points, an error will be raised.

    Gridding objects like :class:`mpol.gridding.DirtyImager` will naturally augment an input dataset to include the Hermitian pairs, so that images of :math:`I_\nu` produced with the inverse Fourier transform turn out to be real.

    Args:
        uu (numpy array): array of u spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
        vv (numpy array): array of v spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
        data (numpy complex): array of data values
        test_vis (int): the number of data points to search for Hermitian 'matches'
        test_channel (int): the index of the channel to perform the check

    Returns:
        boolean : ``True`` if dataset does contain Hermitian pairs, ``False`` if not.
    """

    # make sure everything is in (nchan, nvis) format, to make our lives easier
    if uu.ndim == 1:
        uu = np.atleast_2d(uu)
        vv = np.atleast_2d(vv)
        data = np.atleast_2d(data)

    # but only test a single-channel
    uu = uu[test_channel]
    vv = vv[test_channel]
    data = data[test_channel]

    # if the dataset contains Hermitian pairs, then there will be a large number of visibilities that have matching
    # (uu, vv) and conjugate data values

    # We don't know what order uu or vv might have been augmented in, or sorted after the fact, so we can't
    # rely on quick differencing operations

    num_pairs = 0

    # make uv array same shape as data (nvis, 2)
    uu_vv = np.array([uu, vv]).T  # (nvis, 2)

    for i in range(test_vis):
        # we will approach this as a sort operation.

        # choose a u,v point
        uv_point = uu_vv[i]

        # see if its conjugate exists in the full array
        # nonzero returns a tuple of (2, found_vis)
        # we only need the first dimension, not the u_v dimension
        ind = np.nonzero(uu_vv == -uv_point)[0]

        # if we found nothing, move on to checking the next point
        if ind.size == 0:
            continue

        # if we found something, then take the first result
        ind = ind[0]

        # test to see whether the data is a conjugate
        if data[i] == np.conj(data[ind]):
            num_pairs += 1

    if num_pairs == 0:
        return

    if num_pairs == test_vis:
        raise DataError(
            "Hermitian pairs were found in the data. Please provide data without Hermitian pairs."
        )
    else:
        raise DataError(
            f"{num_pairs} Hermitian pairs were found out of {test_vis} visibilities tested, dataset is inconsistent."
        )

    # choose a uu, vv point, then see if the opposite value exists in the dataset
    # if it does, then check that its visibility value is the complex conjugate

    # we could have a max threshold, i.e., like at least 5 need to exist to say the dataset has pairs

    # Subtract
    return False


class GridderBase:
    r"""
    This class is not designed to be used directly, but rather to be subclassed.

    Subclasses will need to implement a `_grid_visibilities(self,...)` method.

    The GridderBase object uses desired image dimensions (via the ``cell_size`` and ``npix`` arguments) to define a corresponding Fourier plane grid as a :class:`.GridCoords` object. A pre-computed :class:`.GridCoords` can be supplied in lieu of ``cell_size`` and ``npix``, but all three arguments should never be supplied at once. For more details on the properties of the grid that is created, see the :class:`.GridCoords` documentation.

    Subclasses will accept "loose" *ungridded* visibility data and store the arrays to the object as instance attributes. The input visibility data should be the set of visibilities over the full :math:`[-u,u]` and :math:`[-v,v]` domain, and should not contain Hermitian pairs (an error will be raised, if they are encountered).  The visibilities can be 1d for a single continuum channel, or 2d for an image cube. If 1d, visibilities will be converted to 2d arrays of shape ``(1, nvis)``. Like the :class:`~mpol.images.ImageCube` class, after construction, GridderBase assumes that you are operating with a multi-channel set of visibilities. These routines will still work with single-channel 'continuum' visibilities, they will just have nchan = 1 in the first dimension of most products.

    Args:
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        uu (numpy array): (nchan, nvis) array of u spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
        vv (numpy array): (nchan, nvis) array of v spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
        weight (2d numpy array): (nchan, nvis) array of thermal weights. Units of [:math:`1/\mathrm{Jy}^2`]
        data_re (2d numpy array): (nchan, nvis) array of the real part of the visibility measurements. Units of [:math:`\mathrm{Jy}`]
        data_im (2d numpy array): (nchan, nvis) array of the imaginary part of the visibility measurements. Units of [:math:`\mathrm{Jy}`]

    """

    def __init__(
        self,
        coords=None,
        uu=None,
        vv=None,
        weight=None,
        data_re=None,
        data_im=None,
    ):
        # check everything should be 2d, expand if not
        # also checks data does not contain Hermitian pairs
        uu, vv, weight, data_re, data_im = _check_data_inputs_2d(
            uu, vv, weight, data_re, data_im
        )

        # setup the coordinates object
        self.coords = coords
        self.nchan = len(uu)

        # make sure we still fit into the grid
        self.coords.check_data_fit(uu, vv)

        # classes that inherit this will need to set data attributes
        # deciding on whether to include Hermitian pairs
        self.uu = uu
        self.vv = vv
        self.weight = weight
        self.data_re = data_re
        self.data_im = data_im

        # and register cell indices against data
        self._create_cell_indices()

    @classmethod
    def from_image_properties(
        cls,
        cell_size,
        npix,
        uu=None,
        vv=None,
        weight=None,
        data_re=None,
        data_im=None,
    ) -> GridderBase:
        coords = GridCoords(cell_size, npix)
        return cls(coords, uu, vv, weight, data_re, data_im)

    def _create_cell_indices(self):
        # figure out which visibility cell each datapoint lands in, so that
        # we can later assign it the appropriate robust weight for that cell
        # do this by calculating the nearest cell index [0, N] for all samples
        self.index_u = np.array(
            [np.digitize(u_chan, self.coords.u_edges) - 1 for u_chan in self.uu]
        )

        self.index_v = np.array(
            [np.digitize(v_chan, self.coords.v_edges) - 1 for v_chan in self.vv]
        )

    def _sum_cell_values_channel(self, uu, vv, values=None):
        r"""
        Given a list of loose visibility points :math:`(u,v)` and their corresponding values :math:`x`,
        partition the points up into 2D :math:`u-v` cells defined by the ``coords`` object attached to
        the gridder, such that ``cell[i,j]`` has bounds between ``coords.u_edges[j, j+1]`` and ``coords.v_edges[i, i+1]``.
        Then, sum the corresponding values for each :math:`(u,v)` point that falls within each cell. The resulting
        cell value is

        .. math::

            \mathrm{result}_{i,j} = \sum_k \mathrm{values}_k

        where :math:`k` indexes all :math:`(u,v)` points that fall within ``coords.u_edges[j, j+1]`` and ``coords.v_edges[i, i+1]``. In the case that all values are :math:`1`, the result is the number of visibilities within each cell (i.e., a histogram).

        Args:
            uu (np.array): 1D array of East-West spatial frequency coordinates for a specific channel. Units of [:math:`\mathrm{k}\lambda`]
            vv (np.array): 1D array of North-South spatial frequency coordinates for a specific channel. Units of [:math:`\mathrm{k}\lambda`]
            values (np.array): 1D array of values (the same length as uu and vv) to use in the sum over each cell. The default (``values=None``) corresponds to using ``values=np.ones_like(uu)`` such that the routine is equivalent to a histogram.

        Returns:
            A 2D array of size ``(npix, npix)`` in ground format containing the summed cell quantities.
        """

        result = fast_hist.histogram2d(
            vv,
            uu,
            bins=self.coords.ncell_u,
            range=[
                [self.coords.v_bin_min, self.coords.v_bin_max],
                [self.coords.u_bin_min, self.coords.u_bin_max],
            ],
            weights=values,
        )

        # only return the "H" value
        return result

    def _sum_cell_values_cube(self, values=None):
        r"""
        Perform the :func:`~mpol.gridding.DataAverager.sum_cell_values_channel` routine over all channels of the
        input visibilities.

        Args:
            values (iterable): ``(nchan, nvis)`` array of values to use in the sum over each cell. The default (``values=None``) corresponds to using ``values=np.ones_like(uu)`` such that the routine is equivalent to a histogram.

        Returns:
            A 3D array of size ``(nchan, npix, npix)`` in ground format containing the summed cell quantities.

        """
        # calculate the histogrammed result for all channels
        cube = np.empty(
            (self.nchan, self.coords.ncell_v, self.coords.ncell_u),
            dtype="float",
        )

        if values is None:
            # pass None to every channel
            values = [None] * self.nchan

        for i in range(self.nchan):
            cube[i] = self._sum_cell_values_channel(self.uu[i], self.vv[i], values[i])

        return cube

    def _extract_gridded_values_to_loose(self, gridded_quantity):
        r"""
        Extract the gridded cell quantity corresponding to each of the loose visibilities.

        Args:
            A 3D array of size ``(nchan, npix, npix)`` in ground format containing the gridded cell quantities.

        Returns:
            A ``(nchan, nvis)`` array of values corresponding to the loose visibilities, using the quantity in that cell.
        """

        return np.array(
            [
                gridded_quantity[i][self.index_v[i], self.index_u[i]]
                for i in range(self.nchan)
            ]
        )

    def _estimate_cell_standard_deviation(self):
        r"""
        Estimate the `standard deviation <https://en.wikipedia.org/wiki/Standard_deviation>`__ of the real and imaginary visibility values within each :math:`u,v` cell (:math:`\mathrm{cell}_{i,j}`) defined by ``self.coords`` using the following steps.

        1. Calculate the mean real :math:`\mu_\Re` and imaginary :math:`\mu_\Im` values within each cell using a weighted mean, assuming that the visibility function is constant across the cell.
        2. For each visibility :math:`k` that falls within the cell, calculate the real and imaginary residuals (:math:`r_\Re` and :math:`r_\Im`) in units of :math:`\sigma_k`, where :math:`\sigma_k = \sqrt{1/w_k}` and :math:`w_k` is the weight of that visibility.
        3. Calculate the standard deviation :math:`s_{i,j}` of the residual distributions within each cell

        .. math::

            s_{i,j} = \sqrt{\frac{1}{N} \sum_k \left (\sigma_k - \bar{\sigma}_{i,j} \right )^2}

        where :math:`\bar{\sigma}_{i,j}` is first estimated as

        .. math::

            \bar{\sigma}_{i,j} = \frac{1}{N} \sum_k \sigma_k


        Returns:
            std_real, std_imag: two 3D arrays of size ``(nchan, npix, npix)`` in ground format containing the standard deviation of the real and imaginary values within each cell, in units of :math:`\sigma`. If everything is correctly calibrated, we expect :math:`s_{i,j} \approx 1 \forall i,j`.

        """

        # 1. use the gridding routine to calculate the mean real and imaginary values on the grid
        self._grid_visibilities()

        # convert grid back to ground format
        mu_re_gridded = np.fft.fftshift(self.data_re_gridded, axes=(1, 2))
        mu_im_gridded = np.fft.fftshift(self.data_im_gridded, axes=(1, 2))

        # extract the real and imaginary values corresponding to the "loose" visibilities
        # mu_re_gridded and mu_im_gridded are arrays with shape (nchan, ncell_v, ncell_u)
        # self.index_v, self.index_u are (nchan, nvis)
        # we want mu_re and mu_im to be (nchan, nvis)
        mu_re = self._extract_gridded_values_to_loose(mu_re_gridded)
        mu_im = self._extract_gridded_values_to_loose(mu_im_gridded)

        # 2. calculate the real and imaginary residuals for the loose visibilities
        # 1/sigma = np.sqrt(weight)
        residual_re = (self.data_re - mu_re) * np.sqrt(self.weight)
        residual_im = (self.data_im - mu_im) * np.sqrt(self.weight)

        # 3. calculate the standard deviation of the residual visibilities

        # 3.1 first calculate the mean residuals
        # calculate the number of visibilities with each cell
        nvis_cell_grid = self._sum_cell_values_cube()
        # extract this out as a quantity for each visibility
        nvis_cell_loose = self._extract_gridded_values_to_loose(nvis_cell_grid)

        # calculate the mean residuals
        # sum residual values
        bar_sigma_re = self._sum_cell_values_cube(residual_re / nvis_cell_loose)
        bar_sigma_im = self._sum_cell_values_cube(residual_im / nvis_cell_loose)
        # extract back to loose
        bar_sigma_re_loose = self._extract_gridded_values_to_loose(bar_sigma_re)
        bar_sigma_im_loose = self._extract_gridded_values_to_loose(bar_sigma_im)

        # 3.2 calculate the standard deviation of the residuals
        s_re = np.sqrt(
            self._sum_cell_values_cube(
                (residual_re - bar_sigma_re_loose) ** 2 / nvis_cell_loose
            )
        )
        s_im = np.sqrt(
            self._sum_cell_values_cube(
                (residual_im - bar_sigma_im_loose) ** 2 / nvis_cell_loose
            )
        )

        return s_re, s_im

    def _check_scatter_error(self, max_scatter=1.2):
        """
        Checks/compares visibility scatter to a given threshold value ``max_scatter`` and raises an AssertionError if the median scatter across all cells exceeds ``max_scatter``.

        Args:
            max_scatter (float): the maximum permissible scatter in units of standard deviation.

        Returns:
            a dictionary containing keys ``return_status``, ``median_re``, and ``median_im``. ``return_status`` is a boolean that is ``False`` if scatter is within acceptable limits of max_scatter (good), and is ``True`` if scatter exceeds acceptable limits. ``median_re`` and ``median_im`` are the median scatter values returned across all cells, in units of standard deviation (estimated from the provided weights).

        """
        s_re, s_im = self._estimate_cell_standard_deviation()

        median_re = np.median(s_re[s_re > 0])
        median_im = np.median(s_im[s_im > 0])

        return_status = (median_re > max_scatter) or (median_im > max_scatter)

        return {
            "return_status": return_status,
            "median_re": median_re,
            "median_im": median_im,
        }

    def _fliplr_cube(self, cube):
        return cube[:, :, ::-1]

    @property
    def ground_cube(self):
        r"""
        The visibility FFT cube fftshifted for plotting with ``imshow``.

        Returns:
            (torch.complex tensor, of shape ``(nchan, npix, npix)``): the FFT of the image cube, in sky plane format.
        """

        return np.fft.fftshift(self.vis_gridded, axes=(1, 2))


class DataAverager(GridderBase):
    r"""
    The DataAverager object uses desired image dimensions (via the ``cell_size`` and ``npix`` arguments) to define a corresponding Fourier plane grid as a :class:`.GridCoords` object. A pre-computed :class:`.GridCoords` can be supplied in lieu of ``cell_size`` and ``npix``, but all three arguments should never be supplied at once. For more details on the properties of the grid that is created, see the :class:`.GridCoords` documentation.

    The :class:`.DataAverager` object accepts "loose" *ungridded* visibility data and stores the arrays to the object as instance attributes. The input visibility data should be the set of visibilities over the full :math:`[-u,u]` and :math:`[-v,v]` domain, and should not contain Hermitian pairs (an error will be raised, if they are encountered). (Note that, unlike :class:`~mpol.gridding.DirtyImager`, this class *will not* augment the dataset to include Hermitian pairs. This is by design, since Hermitian pairs should not be used in likelihood calculations).

    The input visibilities can be 1d for a single continuum channel, or 2d for image cube. If 1d, visibilities will be converted to 2d arrays of shape ``(1, nvis)``. Like the :class:`~mpol.images.ImageCube` class, after construction, the DataAverager assumes that you are operating with a multi-channel set of visibilities. These routines will still work with single-channel 'continuum' visibilities, they will just have nchan = 1 in the first dimension of most products.

    Once the DataAverager object is initialized with loose visibilities, you can average them and export them for use in Regularized Maximum Likelihood imaging with the :func:`mpol.gridding.DataAverager.to_pytorch_dataset` routine.

    Example::

        averager = gridding.DataAverager(
            coords=coords,
            uu=uu,
            vv=vv,
            weight=weight,
            data_re=data_re,
            data_im=data_im,
            )

        dset = averager.to_pytorch_dataset()


    Args:
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        uu (numpy array): (nchan, nvis) array of u spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
        vv (numpy array): (nchan, nvis) array of v spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
        weight (2d numpy array): (nchan, nvis) array of thermal weights. Units of [:math:`1/\mathrm{Jy}^2`]
        data_re (2d numpy array): (nchan, nvis) array of the real part of the visibility measurements. Units of [:math:`\mathrm{Jy}`]
        data_im (2d numpy array): (nchan, nvis) array of the imaginary part of the visibility measurements. Units of [:math:`\mathrm{Jy}`]

    """

    def _grid_visibilities(self):
        r"""
        Average the loose data visibilities to the Fourier grid.
        """

        # create the cells as edges around the existing points
        # note that at this stage, the UV grid is strictly increasing
        # when in fact, later on, we'll need to fftshift for the FFT
        cell_weight = self._sum_cell_values_cube(self.weight)

        # boolean index for cells that *contain* visibilities
        mask = cell_weight > 0.0

        # calculate the density weights under "uniform"
        # the density weights have the same shape as the re, im samples.
        # cell_weight is (nchan, ncell_v, ncell_u)
        # self.index_v, self.index_u are (nchan, nvis)
        # we want density_weights to be (nchan, nvis)
        density_weight = 1 / self._extract_gridded_values_to_loose(cell_weight)

        # grid the reals and imaginaries separately
        # outputs from _sum_cell_values_cube are *not* pre-packed
        data_re_gridded = self._sum_cell_values_cube(
            self.data_re * density_weight * self.weight
        )

        data_im_gridded = self._sum_cell_values_cube(
            self.data_im * density_weight * self.weight
        )

        # store the pre-packed FFT products for access by outside routines
        self.mask = np.fft.fftshift(mask, axes=(1, 2))
        self.data_re_gridded = np.fft.fftshift(data_re_gridded, axes=(1, 2))
        self.data_im_gridded = np.fft.fftshift(data_im_gridded, axes=(1, 2))
        self.vis_gridded = self.data_re_gridded + self.data_im_gridded * 1.0j

    def _grid_weights(self):
        r"""
        Average the visibility weights to the Fourier grid contained in ``self.coords``, such that
        the ``self.weight_gridded`` corresponds to the equivalent weight on the averaged visibilities
        within that cell.
        """

        # create the cells as edges around the existing points
        # note that at this stage, the UV grid is strictly increasing
        # when in fact, later on, we'll need to fftshift for the FFT
        cell_weight = self._sum_cell_values_cube(self.weight)

        # instantiate uncertainties for each averaged visibility.
        self.weight_gridded = np.fft.fftshift(cell_weight, axes=(1, 2))

    def to_pytorch_dataset(self, check_visibility_scatter=True, max_scatter=1.2):
        r"""
        Export gridded visibilities to a PyTorch dataset object.

        Args:
            check_visibility_scatter (bool): whether the routine should check the standard deviation of visibilities in each within each :math:`u,v` cell (:math:`\mathrm{cell}_{i,j}`) defined by ``self.coords``. Default is ``True``. A ``RuntimeError`` will be raised if any cell has a scatter larger than ``max_scatter``.
            max_scatter (float): the maximum allowable standard deviation of visibility values in a given :math:`u,v` cell (:math:`\mathrm{cell}_{i,j}`) defined by ``self.coords``. Defaults to a factor of 120%.

        Returns:
            :class:`~mpol.datasets.GriddedDataset` with gridded visibilities.
        """

        # check the visibility scatter and flag user if there are issues
        if check_visibility_scatter:
            d = self._check_scatter_error(max_scatter)
            if d["return_status"]:
                raise RuntimeError(
                    "Visibility scatter exceeds ``max_scatter``:{:}, indicating a potential problem with data weights. Consider inspecting weights using CASA tools before exporting visibilities for use with MPoL. Median real scatter: {:} x sigma. Median imag scatter: {:} x sigma.".format(
                        max_scatter, d["median_re"], d["median_im"]
                    )
                )

        # grid visibilities (uniform weighting necessary here) and weights
        self._grid_visibilities()
        self._grid_weights()

        return GriddedDataset(
            coords=self.coords,
            nchan=self.nchan,
            vis_gridded=self.vis_gridded,
            weight_gridded=self.weight_gridded,
            mask=self.mask,
        )


class DirtyImager(GridderBase):
    r"""
    This class is mainly used for producing diagnostic "dirty" images of the visibility data.

    The DirtyImager object uses desired image dimensions (via the ``cell_size`` and ``npix`` arguments) to define a corresponding Fourier plane grid as a :class:`.GridCoords` object. A pre-computed :class:`.GridCoords` can be supplied in lieu of ``cell_size`` and ``npix``, but all three arguments should never be supplied at once. For more details on the properties of the grid that is created, see the :class:`.GridCoords` documentation.

    The :class:`.DirtyImager` object accepts "loose" *ungridded* visibility data and stores the arrays to the object as instance attributes. The input visibility data should be the normal set of visibilities over the full :math:`[-u,u]` and :math:`[-v,v]` domain; internally the DirtyImager will automatically augment the dataset to include the complex conjugates, i.e. the 'Hermitian pairs.'

    The input visibilities can be 1d for a single continuum channel, or 2d for image cube. If 1d, visibilities will be converted to 2d arrays of shape ``(1, nvis)``. Like the :class:`~mpol.images.ImageCube` class, after construction, the DirtyImager assumes that you are operating with a multi-channel set of visibilities. These routines will still work with single-channel 'continuum' visibilities, they will just have nchan = 1 in the first dimension of most products.

    Example::

        imager = gridding.DirtyImager(
            coords=coords,
            uu=uu,
            vv=vv,
            weight=weight,
            data_re=data_re,
            data_im=data_im,
            )

        img, beam = imager.get_dirty_image(weighting="briggs", robust=0.0)


    Args:
        cell_size (float): width of a single square pixel in [arcsec]
        npix (int): number of pixels in the width of the image
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        uu (numpy array): (nchan, nvis) array of u spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
        vv (numpy array): (nchan, nvis) array of v spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
        weight (2d numpy array): (nchan, nvis) array of thermal weights. Units of [:math:`1/\mathrm{Jy}^2`]
        data_re (2d numpy array): (nchan, nvis) array of the real part of the visibility measurements. Units of [:math:`\mathrm{Jy}`]
        data_im (2d numpy array): (nchan, nvis) array of the imaginary part of the visibility measurements. Units of [:math:`\mathrm{Jy}`]



    """

    def __init__(
        self,
        coords=None,
        uu=None,
        vv=None,
        weight=None,
        data_re=None,
        data_im=None,
    ):
        # check everything should be 2d, expand if not
        # also checks data does not contain Hermitian pairs
        uu, vv, weight, data_re, data_im = _check_data_inputs_2d(
            uu, vv, weight, data_re, data_im
        )

        # setup the coordinates object
        self.coords = coords
        self.nchan = len(uu)

        # make sure we still fit into the grid
        self.coords.check_data_fit(uu, vv)

        # expand the vectors to include complex conjugates
        uu_full = np.concatenate([uu, -uu], axis=1)
        vv_full = np.concatenate([vv, -vv], axis=1)

        # make sure we still fit into the grid (with expansion)
        self.coords.check_data_fit(uu_full, vv_full)

        self.uu = uu_full
        self.vv = vv_full
        self.weight = np.concatenate([weight, weight], axis=1)
        self.data_re = np.concatenate([data_re, data_re], axis=1)
        self.data_im = np.concatenate([data_im, -data_im], axis=1)

        # and register cell indices against data
        self._create_cell_indices()

    def _grid_visibilities(
        self,
        weighting="uniform",
        robust=None,
        taper_function=None,
    ):
        r"""
        Grid the loose data visibilities to the Fourier grid in preparation for imaging.

        Args:
            weighting (string): The type of cell averaging to perform. Choices of ``"natural"``, ``"uniform"``, or ``"briggs"``, following CASA tclean. If ``"briggs"``, also specify a robust value.
            robust (float): If ``weighting='briggs'``, specify a robust value in the range [-2, 2]. ``robust=-2`` approximately corresponds to uniform weighting and ``robust=2`` approximately corresponds to natural weighting.
            taper_function (function reference): a function assumed to be of the form :math:`f(u,v)` which calculates a prefactor in the range :math:`[0,1]` and premultiplies the visibility data. The function must assume that :math:`u` and :math:`v` will be supplied in units of :math:`\mathrm{k}\lambda`. By default no taper is applied.
        """

        if taper_function is None:
            tapering_weight = np.ones_like(self.weight)
        else:
            tapering_weight = taper_function(self.uu, self.vv)

        # create the cells as edges around the existing points
        # note that at this stage, the UV grid is strictly increasing
        # when in fact, later on, we'll need to fftshift for the FFT
        cell_weight = self._sum_cell_values_cube(self.weight)

        # boolean index for cells that *contain* visibilities
        mask = cell_weight > 0.0

        # calculate the density weights
        # the density weights have the same shape as the re, im samples.
        if weighting == "natural":
            density_weight = np.ones_like(self.weight)
        elif weighting == "uniform":
            # cell_weight is (nchan, ncell_v, ncell_u)
            # self.index_v, self.index_u are (nchan, nvis)
            # we want density_weights to be (nchan, nvis)
            density_weight = 1 / self._extract_gridded_values_to_loose(cell_weight)

        elif weighting == "briggs":
            if robust is None or not -2 <= robust <= 2:
                raise ValueError(
                    "With 'briggs' weighting, a robust value must be specified between [-2, 2]."
                )

            # implement robust weighting using the definition used in CASA
            # https://casa.nrao.edu/casadocs-devel/stable/imaging/synthesis-imaging/data-weighting

            # calculate the robust parameter f^2 for each channel
            f_sq = ((5 * 10 ** (-robust)) ** 2) / (
                np.sum(cell_weight**2, axis=(1, 2)) / np.sum(self.weight, axis=1)
            )

            # the robust weight corresponding to the cell
            cell_robust_weight = 1 / (1 + cell_weight * f_sq[:, np.newaxis, np.newaxis])

            # zero out cells that have no visibilities
            # to prevent normalization error in next step
            cell_robust_weight[~mask] = 0

            # now assign the cell robust weight to each visibility within that cell
            density_weight = self._extract_gridded_values_to_loose(cell_robust_weight)

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
        # outputs from _sum_cell_values_cube are *not* pre-packed
        data_re_gridded = self._sum_cell_values_cube(
            self.data_re * tapering_weight * density_weight * self.weight
        )

        data_im_gridded = self._sum_cell_values_cube(
            self.data_im * tapering_weight * density_weight * self.weight
        )

        # the beam is the response to a point source, which is data_re = constant, data_im = 0
        # so we save time and only calculate the reals, because gridded_beam_im = 0
        re_gridded_beam = self._sum_cell_values_cube(
            tapering_weight * density_weight * self.weight
        )

        # store the pre-packed FFT products for access by outside routines
        self.mask = np.fft.fftshift(mask)
        self.data_re_gridded = np.fft.fftshift(data_re_gridded, axes=(1, 2))
        self.data_im_gridded = np.fft.fftshift(data_im_gridded, axes=(1, 2))
        self.vis_gridded = self.data_re_gridded + self.data_im_gridded * 1.0j
        self.re_gridded_beam = np.fft.fftshift(re_gridded_beam, axes=(1, 2))

    def _get_dirty_beam(self, C, re_gridded_beam):
        """
        Compute the dirty beam corresponding to the gridded visibilities.

        Args:
            C (1D np.array): normalization constants for each channel
            re_gridded_beam (3d np.array): the gridded visibilities corresponding to a unit point source in the center of the field.

        Returns:
            numpy image cube with a dirty beam (PSF) for each channel. By definition, the peak is normalized to 1.0.
        """
        # if we're sticking to the dirty beam and image equations in Briggs' Ph.D. thesis,
        # no correction for du or dv prefactors needed here
        # that is because we are using the FFT to compute an already discretized equation, not
        # approximating a continuous equation.

        beam = self._fliplr_cube(
            np.fft.fftshift(
                self.coords.npix**2
                * np.fft.ifft2(
                    C[:, np.newaxis, np.newaxis] * re_gridded_beam,
                ),
                axes=(1, 2),
            )
        )

        if np.max(beam.imag) >= 1e-10:
            raise ThresholdExceededError(
                "Dirty beam contained substantial imaginary values, check input visibilities, otherwise raise a github issue."
            )

        self.beam = beam.real

        return self.beam

    def _null_dirty_beam(self, ntheta=24, single_channel_estimate=True):
        r"""Zero out (null) all pixels in the dirty beam exterior to the first null, for each channel.

        Args:
            ntheta (int): number of azimuthal wedges to use for the 1st null calculation. More wedges will result in a more accurate estimate of dirty beam area, but will also take longer.
            single_channel_estimate (bool): If ``True`` (the default), use the area estimated from the first channel for all channels in the multi-channel image cube. If ``False``, calculate the beam area for all channels.

        Returns: a cube like the dirty beam, but with all pixels exterior to the first null set to 0.
        """

        try:
            self.beam
        except AttributeError:
            self._get_dirty_beam(self.C, self.re_gridded_beam)

        # consider the 2D beam for each channel described by polar coordinates r, theta.
        #
        # this routine works by finding the smallest r for which the beam goes negative (the first null)
        # as a function of theta. Then, for this same theta, all pixels (negative or not) with values of r larger than
        # this are set to 0.

        # the end product of this routine will be a "nulled" beam, which can be used in the calculation
        # of dirty beam area.

        # the angular extent for each "slice"
        # the smaller the slice, the more accurate the area estimate, but also the
        # longer it takes
        da = 2 * np.pi / ntheta  # radians
        azimuths = np.arange(0, 2 * np.pi, da)

        # calculate a meshgrid (same for all channels)
        ll, mm = np.meshgrid(self.coords.l_centers, self.coords.m_centers)
        rr = np.sqrt(ll**2 + mm**2)
        theta = np.arctan2(mm, ll) + np.pi  # radians in range [0, 2pi]

        nulled_beam = self.beam.copy()
        # for each channel,
        # find the first occurrence of a non-zero value, such that we end up with a continuous
        # ring of masked values.
        for i in range(self.nchan):
            nb = nulled_beam[i]
            ind_neg = nb < 0

            for a in azimuths:
                # examine values between a, a+da with some overlap
                ind_azimuth = (theta >= a - 0.3 * da) & (theta <= (a + 1.3 * da))

                # find all negative values within azimuth slice
                ind_neg_and_az = ind_neg & ind_azimuth

                # find the smallest r within this slice
                min_r = np.min(rr[ind_neg_and_az])

                # null all pixels within this slice with radii r or greater
                ind_r = rr >= min_r
                ind_r_and_az = ind_r & ind_azimuth
                nb[ind_r_and_az] = 0

            if single_channel_estimate:
                # just copy the mask from the first channel to all channels
                ind_0 = nb == 0
                nulled_beam[:, ind_0] = 0
                break

        return nulled_beam

    def get_dirty_beam_area(self, ntheta=24, single_channel_estimate=True):
        r"""
        Compute the effective area of the dirty beam for each channel. Assumes that the beam has already been generated by running :func:`~mpol.gridding.DirtyImager.get_dirty_image`. This is an approximate calculation involving a simple sum over all pixels out to the first null (zero crossing) of the dirty beam. This quantity is designed to approximate the conversion of image units from :math:`[\mathrm{Jy}\,\mathrm{beam}^{-1}]` to :math:`[\mathrm{Jy}\,\mathrm{arcsec}^{-2}]`, even though units of :math:`[\mathrm{Jy}\,\mathrm{dirty\;beam}^{-1}]` are technically undefined.

        Args:
            ntheta (int): number of azimuthal wedges to use for the 1st null calculation. More wedges will result in a more accurate estimate of dirty beam area, but will also take longer.
            single_channel_estimate (bool): If ``True`` (the default), use the area estimated from the first channel for all channels in the multi-channel image cube. If ``False``, calculate the beam area for all channels.

        Returns:
            (1D numpy array float) beam area for each channel in units of :math:`[\mathrm{arcsec}^{2}]`
        """
        nulled = self._null_dirty_beam(
            ntheta=ntheta, single_channel_estimate=single_channel_estimate
        )
        return self.coords.cell_size**2 * np.sum(nulled, axis=(1, 2))  # arcsec^2

    def get_dirty_image(
        self,
        weighting="uniform",
        robust=None,
        taper_function=None,
        unit="Jy/beam",
        check_visibility_scatter=True,
        max_scatter=1.2,
        **beam_kwargs,
    ):
        r"""
        Calculate the dirty image.

        Args:
            weighting (string): The type of cell averaging to perform. Choices of ``"natural"``, ``"uniform"``, or ``"briggs"``, following CASA tclean. If ``"briggs"``, also specify a robust value.
            robust (float): If ``weighting='briggs'``, specify a robust value in the range [-2, 2]. ``robust=-2`` approxmately corresponds to uniform weighting and ``robust=2`` approximately corresponds to natural weighting.
            taper_function (function reference): a function assumed to be of the form :math:`f(u,v)` which calculates a prefactor in the range :math:`[0,1]` and premultiplies the visibility data. The function must assume that :math:`u` and :math:`v` will be supplied in units of :math:`\mathrm{k}\lambda`. By default no taper is applied.
            unit (string): what unit should the image be in. Default is ``"Jy/beam"``. If ``"Jy/arcsec^2"``, then the effective area of the dirty beam will be used to convert from ``"Jy/beam"`` to ``"Jy/arcsec^2"``.
            check_visibility_scatter (bool): whether the routine should check the standard deviation of visibilities in each within each :math:`u,v` cell (:math:`\mathrm{cell}_{i,j}`) defined by ``self.coords``. Default is ``True``. A ``RuntimeWarning`` will be raised if any cell has a scatter larger than ``max_scatter``.
            max_scatter (float): the maximum allowable standard deviation of visibility values in a given :math:`u,v` cell (:math:`\mathrm{cell}_{i,j}`) defined by ``self.coords``. Defaults to a factor of 120%.
            **beam_kwargs: all additional keyword arguments passed to :func:`~mpol.gridding.get_dirty_beam_area` if ``unit="Jy/arcsec^2"``.

        Returns:
            2-tuple of (``image``, ``beam``) where ``image`` is an (nchan, npix, npix) numpy array of the dirty image cube in units ``unit``. ``beam`` is an numpy image cube with a dirty beam (PSF) for each channel. The units of the beam are always Jy/{dirty beam}, i.e., the peak of the beam is normalized to 1.0.
        """

        # check unit input
        if unit not in ["Jy/beam", "Jy/arcsec^2"]:
            raise ValueError("Unknown unit", unit)

        # check the visibility scatter and flag user if there are issues
        if check_visibility_scatter:
            d = self._check_scatter_error(max_scatter)
            if d["return_status"]:
                warnings.warn(
                    RuntimeWarning(
                        "Visibility scatter exceeds ``max_scatter``:{:}, indicating a potential problem with data weights. Consider inspecting weights using CASA tools before exporting visibilities for use with MPoL. Median real scatter: {:} x sigma. Median imag scatter: {:} x sigma.".format(
                            max_scatter, d["median_re"], d["median_im"]
                        )
                    )
                )

        # call _grid_visibilities
        # inputs for weighting will be checked inside _grid_visibilities
        self._grid_visibilities(
            weighting=weighting,
            robust=robust,
            taper_function=taper_function,
        )

        img = self._fliplr_cube(
            np.fft.fftshift(
                self.coords.npix**2
                * np.fft.ifft2(self.C[:, np.newaxis, np.newaxis] * self.vis_gridded),
                axes=(1, 2),
            )
        )  # Jy/beam

        # calculate the beam
        # also pre-stores internal self.beam value for area routine, if necessary
        beam = self._get_dirty_beam(self.C, self.re_gridded_beam)

        # for units of Jy/arcsec^2, we could just leave out the C constant *if* we were doing
        # uniform weighting. The relationships get more complex for robust or natural weighting, however,
        # so it's safer to calculate the number of arcseconds^2 per beam
        if unit == "Jy/arcsec^2":
            beam_area_per_chan = self.get_dirty_beam_area(**beam_kwargs)  # [arcsec^2]

            # convert image
            # (Jy/1 arcsec^2) = (Jy/ 1 beam) * (1 beam/ n arcsec^2)
            # beam_area_per_chan is the n of arcsec^2 per 1 beam

            img /= beam_area_per_chan[:, np.newaxis, np.newaxis]

        if np.max(img.imag) >= 1e-10:
            raise ThresholdExceededError(
                "Dirty image contained substantial imaginary values, check input visibilities, otherwise raise a github issue."
            )

        return img.real, beam
