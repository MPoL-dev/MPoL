import copy

import numpy as np
import torch
from torch.utils.data import Dataset

from . import spheroidal_gridding, utils
from .constants import *
from .coordinates import _setup_coords
from .utils import loglinspace


class GriddedDataset:
    r"""
    Args:
        cell_size (float): the width of a pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        nchan (int): the number of channels in the image (default = 1).
        vis_gridded (torch complex): the gridded visibility data stored in a "packed" format (pre-shifted for fft)
        weight_gridded (torch double): the weights corresponding to the gridded visibility data, also in a packed format
        mask (torch boolean): a boolean mask to index the non-zero locations of ``vis_gridded`` and ``weight_gridded`` in their packed format.
        device (torch.device) : the desired device of the dataset. If ``None``, defalts to current device.


    After initialization, the GriddedDataset provides the non-zero cells of the gridded visibilities and weights as a 1D vector via the following instance variables. This means that any individual channel information has been collapsed.

    :ivar vis_indexed: 1D complex tensor of visibility data
    :ivar weight_indexd: 1D tensor of weight values

    If you index the output of the Fourier layer in the same manner using ``self.mask`` (as done internally within :class:`~mpol.connectors.DataConnector`), then the model and data visibilities can be directly compared using a loss function.
    """

    def __init__(
        self,
        cell_size=None,
        npix=None,
        coords=None,
        nchan=None,
        vis_gridded=None,
        weight_gridded=None,
        mask=None,
        device=None,
    ):

        _setup_coords(self, cell_size, npix, coords, nchan)

        self.vis_gridded = torch.tensor(vis_gridded, device=device)
        self.weight_gridded = torch.tensor(weight_gridded, device=device)
        self.mask = torch.tensor(mask, device=device)

        # pre-index the values
        # note that these are *collapsed* across all channels
        # 1D array
        self.vis_indexed = self.vis_gridded[self.mask]
        self.weight_indexed = self.weight_gridded[self.mask]

    def add_mask(self, mask, device=None):
        r"""
        Apply an additional mask to the data. Only works as a data limiting operation (i.e., ``mask`` is more restrictive than the mask already attached to the dataset).

        Args:
            mask (2D numpy or PyTorch tensor): boolean mask (in packed format) to apply to dataset. Assumes input will be broadcast across all channels. Mask must be Hermitian, like the visibilities themselves.
            device (torch.device) : the desired device of the dataset. If ``None``, defalts to current device.
        """

        new_2D_mask = torch.tensor(mask, device=device)
        new_3D_mask = torch.broadcast_to(new_2D_mask, self.mask.size())

        # update mask via an AND operation, meaning we will only keep visibilities that are
        # 1) part of the original dataset
        # 2) valid within the new mask
        self.mask = torch.logical_and(self.mask, new_3D_mask)

        # zero out vis_gridded and weight_gridded that may have existed
        # but are no longer valid
        # These operations on the gridded quantities are only important for routines
        # that grab these quantities directly, like residual grid imager
        self.vis_gridded[~self.mask] = 0.0
        self.weight_gridded[~self.mask] = 0.0

        # update pre-indexed values
        self.vis_indexed = self.vis_gridded[self.mask]
        self.weight_indexed = self.weight_gridded[self.mask]

    @property
    def ground_mask(self):
        r"""
        The boolean mask, arranged in ground format.

        Returns:
            torch.boolean : 3D mask cube of shape ``(nchan, npix, npix)``

        """
        return utils.packed_cube_to_ground_cube(self.mask)

    def to(self, device):
        """
        Moves the tensors of the dataset to specified device.

        Args:
            device (torch.device): the desired device

        Returns:
            copy of the GriddedDataset instance on the new device
        """
        self.vis_gridded = self.vis_gridded.to(device)
        self.weight_gridded = self.weight_gridded.to(device)
        self.mask = self.mask.to(device)

        # pre-index the values
        # note that these are *collapsed* across all channels
        # 1D array
        self.vis_indexed = self.vis_indexed.to(device)
        self.weight_indexed = self.weight_indexed.to(device)

        return self


# custom dataset loader
class UVDataset(Dataset):
    r"""
    Container for loose interferometric visibilities.

    Args:
        uu (2d numpy array): (nchan, nvis) length array of u spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
        vv (2d numpy array): (nchan, nvis) length array of v spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
        data_re (2d numpy array): (nchan, nvis) length array of the real part of the visibility measurements. Units of [:math:`\mathrm{Jy}`]
        data_im (2d numpy array): (nchan, nvis) length array of the imaginary part of the visibility measurements. Units of [:math:`\mathrm{Jy}`]
        weights (2d numpy array): (nchan, nvis) length array of thermal weights. Units of [:math:`1/\mathrm{Jy}^2`]
        cell_size (float): the image pixel size in arcsec. Defaults to None, but if both `cell_size` and `npix` are set, the visibilities will be pre-gridded to the RFFT output dimensions.
        npix (int): the number of pixels per image side (square images only). Defaults to None, but if both `cell_size` and `npix` are set, the visibilities will be pre-gridded to the RFFT output dimensions.
        device (torch.device) : the desired device of the dataset. If ``None``, defalts to current device.

    If both `cell_size` and `npix` are set, the dataset will be automatically pre-gridded to the RFFT output grid. This will greatly speed up performance.

    If you have just a single channel, you can pass 1D numpy arrays for `uu`, `vv`, `weights`, `data_re`, and `data_im` and they will automatically be promoted to 2D with a leading dimension of 1 (i.e., ``nchan=1``).
    """

    def __init__(
        self,
        uu=None,
        vv=None,
        weights=None,
        data_re=None,
        data_im=None,
        cell_size=None,
        npix=None,
        device=None,
        **kwargs
    ):

        # assert that all vectors are the same shape
        shape = uu.shape
        for a in [vv, weights, data_re, data_im]:
            assert a.shape == shape, "All dataset inputs must be the same shape."

        if len(shape) == 1:
            uu = np.atleast_2d(uu)
            vv = np.atleast_2d(vv)
            data_re = np.atleast_2d(data_re)
            data_im = np.atleast_2d(data_im)
            weights = np.atleast_2d(weights)

        self.nchan = shape[0]

        assert np.all(
            weights > 0.0
        ), "Not all thermal weights are positive, check inputs."

        if cell_size is not None and npix is not None:
            self.cell_size = cell_size * arcsec  # [radians]
            self.npix = npix

            (
                uu_grid,
                vv_grid,
                grid_mask,
                g_weights,
                g_re,
                g_im,
            ) = spheroidal_gridding.grid_dataset(
                uu,
                vv,
                weights,
                data_re,
                data_im,
                self.cell_size / arcsec,
                npix=self.npix,
            )

            # grid_mask (nchan, npix, npix//2 + 1) bool: a boolean array the same size as the output of the RFFT, designed to directly index into the output to evaluate against pre-gridded visibilities.
            self.uu = torch.tensor(uu_grid, device=device)
            self.vv = torch.tensor(vv_grid, device=device)
            self.grid_mask = torch.tensor(grid_mask, dtype=torch.bool, device=device)
            self.weights = torch.tensor(g_weights, device=device)
            self.re = torch.tensor(g_re, device=device)
            self.im = torch.tensor(g_im, device=device)
            self.gridded = True

        else:
            self.gridded = False
            self.uu = torch.tensor(uu, dtype=torch.double, device=device)  # klambda
            self.vv = torch.tensor(vv, dtype=torch.double, device=device)  # klambda
            self.weights = torch.tensor(
                weights, dtype=torch.double, device=device
            )  # 1/Jy^2
            self.re = torch.tensor(data_re, dtype=torch.double, device=device)  # Jy
            self.im = torch.tensor(data_im, dtype=torch.double, device=device)  # Jy

        # TODO: store kwargs to do something for antenna self-cal

    def __getitem__(self, index):
        return (
            self.uu[index],
            self.vv[index],
            self.weights[index],
            self.re[index],
            self.im[index],
        )

    def __len__(self):
        return len(self.uu)


class Dartboard:
    r"""
    A polar coordinate grid relative to a :class:`~mpol.coordinates.GridCoords` object, reminiscent of a dartboard layout. The main utility of this object is to support splitting a dataset along radial and azimuthal bins for k-fold cross validation.

    Args:
        cell_size (float): the width of a pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        q_edges (1D numpy array): an array of radial bin edges to set the dartboard cells in :math:`[\mathrm{k}\lambda]`. If ``None``, defaults to 12 log-linearly radial bins stretching from 0 to the :math:`q_\mathrm{max}` represented by ``coords``.
        phi_edges (1D numpy array): an array of azimuthal bin edges to set the dartboard cells in [radians], over the domain :math:`[0, \pi]`, which is also implicitly mapped to the domain :math:`[-\pi, \pi]` to preserve the Hermitian nature of the visibilities. If ``None``, defaults to 8 equal-spaced azimuthal bins stretched from :math:`0` to :math:`\pi`.

    """

    def __init__(
        self, cell_size=None, npix=None, coords=None, q_edges=None, phi_edges=None
    ):

        _setup_coords(self, cell_size, npix, coords)

        # copy over relevant quantities from coords
        # these are in packed format
        self.cartesian_qs = self.coords.packed_q_centers_2D
        self.cartesian_phis = self.coords.packed_phi_centers_2D

        # set q_max to the max q in coords
        self.q_max = self.coords.q_max  # [klambda]

        if q_edges is not None:
            self.q_edges = q_edges
        else:
            # set q edges approximately following inspriation from Petry et al. scheme:
            # https://ui.adsabs.harvard.edu/abs/2020SPIE11449E..1DP/abstract
            # first two bins set to 7m width
            # after third bin, bin width increases linearly until it is 700m at 16km baseline.
            # From 16m to 16km, bin width goes from 7m to 700m.

            # We aren't doing quite the same thing, just logspacing with a few linear cells at the start.
            self.q_edges = loglinspace(0, self.q_max, N_log=8, M_linear=5)

        if phi_edges is not None:
            assert np.all(phi_edges >= 0) & np.all(
                phi_edges <= np.pi
            ), "phi edges must be between 0 and pi"
            self.phi_edges = phi_edges
        else:
            # set phi edges
            self.phi_edges = np.linspace(0, np.pi, num=8 + 1)  # [radians]

    def get_polar_histogram(self, qs, phis):
        r"""
        Calculate a histogram in polar coordinates, using the bin edges defined by ``q_edges`` and ``phi_edges`` during initialization.

        Data coordinates should include the points for the Hermitian visibilities.

        Args:
            qs: 1d array of q values :math:`[\mathrm{k}\lambda]`
            phis: 1d array of datapoint azimuth values [radians] (must be the same length as qs)

        Returns:
            2d integer numpy array of cell counts, i.e., how many datapoints fell into each dartboard cell.

        """

        # make a polar histogram
        H, x_edges, y_edges = np.histogram2d(
            qs, phis, bins=[self.q_edges, self.phi_edges]
        )

        return H

    def get_nonzero_cell_indices(self, qs, phis):
        r"""
        Return a list of the cell indices that contain data points, using the bin edges defined by ``q_edges`` and ``phi_edges`` during initialization.

        Data coordinates should include the points for the Hermitian visibilities.

        Args:
            qs: 1d array of q values :math:`[\mathrm{k}\lambda]`
            phis: 1d array of datapoint azimuth values [radians] (must be the same length as qs)

        Returns:
            list of cell indices where cell contains at least one datapoint.
        """

        # make a polar histogram
        H = self.get_polar_histogram(qs, phis)

        indices = np.argwhere(H > 0)  # [i,j] indexes to go to q, phi

        return indices

    def build_grid_mask_from_cells(self, cell_index_list):
        r"""
        Create a boolean mask of size ``(npix, npix)`` (in packed format) corresponding to the ``vis_gridded`` and ``weight_gridded`` quantities of the :class:`~mpol.datasets.GriddedDataset` .

        Args:
            cell_index_list (list): list or iterable containing [q_cell, phi_cell] index pairs to include in the mask.

        Returns: (numpy array) 2D boolean mask in packed format.
        """
        mask = np.zeros_like(self.cartesian_qs, dtype="bool")

        # uses about a Gb..., and this only 256x256
        for cell_index in cell_index_list:

            qi, pi = cell_index
            q_min, q_max = self.q_edges[qi : qi + 2]
            p0_min, p0_max = self.phi_edges[pi : pi + 2]
            # also include Hermitian values
            p1_min, p1_max = self.phi_edges[pi : pi + 2] - np.pi

            # whether or not the q and phi values of the coordinate array
            # fit in the q cell and *either of* the regular or Hermitian phi cell
            ind = (
                (self.cartesian_qs >= q_min)
                & (self.cartesian_qs < q_max)
                & (
                    ((self.cartesian_phis > p0_min) & (self.cartesian_phis <= p0_max))
                    | ((self.cartesian_phis > p1_min) & (self.cartesian_phis <= p1_max))
                )
            )

            mask[ind] = True

        return mask


class KFoldCrossValidatorGridded:
    r"""
    Split a GriddedDataset into :math:`k` non-overlapping chunks, internally partitioned by a Dartboard. Inherit the properties of the GriddedDataset. This object creates an iterator providing a (train, test) pair of :class:`~mpol.datasets.GriddedDataset` for each k-fold.

    Args:
        griddedDataset (:class:`~mpol.datasets.GriddedDataset`): instance of the gridded dataset
        k (int): the number of subpartitions of the dataset
        dartboard (:class:`~mpol.datasets.Dartboard`): a pre-initialized Dartboard instance. If ``dartboard`` is provided, do not provide ``q_edges`` or ``phi_edges``.
        q_edges (1D numpy array): an array of radial bin edges to set the dartboard cells in :math:`[\mathrm{k}\lambda]`. If ``None``, defaults to 12 log-linearly radial bins stretching from 0 to the :math:`q_\mathrm{max}` represented by ``coords``.
        phi_edges (1D numpy array): an array of azimuthal bin edges to set the dartboard cells in [radians]. If ``None``, defaults to 8 equal-spaced azimuthal bins stretched from :math:`0` to :math:`\pi`.
        npseed (int): (optional) numpy random seed to use for the permutation, for reproducibility

    Once initialized, iterate through the datasets like

    >>> cv = datasets.KFoldCrossValidatorGridded(dataset, k)
    >>> for (train, test) in cv: # iterate among k datasets
    >>> ... # working with the n-th slice of k datasets
    >>> ... # do operations with train dataset
    >>> ... # do operations with test dataset

    """

    def __init__(
        self,
        griddedDataset,
        k,
        dartboard=None,
        q_edges=None,
        phi_edges=None,
        npseed=None,
    ):

        self.griddedDataset = griddedDataset

        assert k > 0, "k must be a positive integer"
        self.k = k

        if dartboard is not None:
            assert (q_edges is None) and (
                phi_edges is None
            ), "If providing a Dartboard instance, do not provide q_edges and phi_edges parameters."
            self.dartboard = dartboard
        else:
            self.dartboard = Dartboard(
                coords=self.griddedDataset.coords, q_edges=q_edges, phi_edges=phi_edges
            )

        # 2D mask for any UV cells that contain visibilities
        # in *any* channel
        stacked_mask = np.any(self.griddedDataset.mask.detach().numpy(), axis=0)

        # get qs, phis from dataset and turn into 1D lists
        qs = self.griddedDataset.coords.packed_q_centers_2D[stacked_mask]
        phis = self.griddedDataset.coords.packed_phi_centers_2D[stacked_mask]

        # create the full cell_list
        self.cell_list = self.dartboard.get_nonzero_cell_indices(qs, phis)

        # partition the cell_list into k pieces
        # first, randomly permute the sequence to make sure
        # we don't get structured radial/azimuthal patterns
        if npseed is not None:
            np.random.seed(npseed)
        self.k_split_cell_list = np.array_split(
            np.random.permutation(self.cell_list), k
        )

    def __iter__(self):
        self.n = 0  # the current k-slice we're on
        return self

    def __next__(self):
        if self.n < self.k:
            k_list = self.k_split_cell_list.copy()
            cell_list_test = k_list.pop(self.n)

            # put the remaining indices back into a full list
            cell_list_train = np.concatenate(k_list)

            # create the masks for each cell_list
            train_mask = self.dartboard.build_grid_mask_from_cells(cell_list_train)
            test_mask = self.dartboard.build_grid_mask_from_cells(cell_list_test)

            # copy original dateset
            train = copy.deepcopy(self.griddedDataset)
            test = copy.deepcopy(self.griddedDataset)

            # and use these masks to limit new datasets to only unmasked cells
            train.add_mask(train_mask)
            test.add_mask(test_mask)

            self.n += 1
            return train, test
        else:
            raise StopIteration
