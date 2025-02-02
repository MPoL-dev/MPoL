from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy import floating, integer
from numpy.typing import ArrayLike, NDArray

from mpol import utils
from mpol.coordinates import GridCoords


class GriddedDataset(torch.nn.Module):
    r"""
    Parameters
    ----------
    coords : :class:`~mpol.coordinates.GridCoords`
        If providing this, cannot provide ``cell_size`` or ``npix``.
    vis_gridded : :class:`torch.Tensor` of :class:`torch.complex128`
        the gridded visibility data stored in a "packed" format (pre-shifted for fft)
    weight_gridded : :class:`torch.Tensor` 
        the weights corresponding to the gridded visibility data,
        also in a packed format
    mask : :class:`torch.Tensor` of :class:`torch.bool`
        a boolean mask to index the non-zero locations of ``vis_gridded`` and
        ``weight_gridded`` in their packed format.
    nchan : int
        the number of channels in the image (default = 1).


    After initialization, the GriddedDataset provides the non-zero cells of the
    gridded visibilities and weights as a 1D vector via the following instance
    variables. This means that any individual channel information has been collapsed.

    :ivar vis_indexed: 1D complex tensor of visibility data

    :ivar weight_indexed: 1D tensor of weight values


    If you index the output of the Fourier layer in the same manner using ``self.mask``,
    then the model and data visibilities can be directly compared using a loss function.
    """

    def __init__(
        self,
        *,
        coords: GridCoords,
        vis_gridded: torch.Tensor,
        weight_gridded: torch.Tensor,
        mask: torch.Tensor,
        nchan: int = 1,
    ) -> None:
        super().__init__()

        self.coords = coords
        self.nchan = nchan

        # store variables as buffers of the module
        self.register_buffer("vis_gridded", vis_gridded)
        self.register_buffer("weight_gridded", weight_gridded)
        self.register_buffer("mask", mask)
        self.vis_gridded: torch.Tensor
        self.weight_gridded: torch.Tensor
        self.mask: torch.Tensor

        # pre-index the values
        # note that these are *collapsed* across all channels
        # 1D array
        self.register_buffer("vis_indexed", self.vis_gridded[self.mask])
        self.register_buffer("weight_indexed", self.weight_gridded[self.mask])
        self.vis_indexed: torch.Tensor
        self.weight_indexed: torch.Tensor

    def add_mask(
        self,
        mask: ArrayLike,
    ) -> None:
        r"""
        Apply an additional mask to the data. Only works as a data limiting operation
        (i.e., ``mask`` is more restrictive than the mask already attached
        to the dataset).

        Args:
            mask (2D numpy or PyTorch tensor): boolean mask (in packed format) to
                apply to dataset. Assumes input will be broadcast across all channels.
        """

        new_2D_mask = torch.Tensor(mask).detach()
        new_3D_mask = torch.broadcast_to(new_2D_mask, self.mask.size())

        # update mask via an AND operation, we will only keep visibilities that are
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

    def forward(self, modelVisibilityCube: torch.Tensor) -> torch.Tensor:
        """
        Args:
            modelVisibilityCube (complex torch.tensor): with shape
                ``(nchan, npix, npix)`` to be indexed. In "pre-packed" format, as in
                output from :meth:`mpol.fourier.FourierCube.forward()`

        Returns:
            torch complex tensor:  1d torch tensor of indexed model samples collapsed
                across cube dimensions.
        """

        assert (
            modelVisibilityCube.size()[0] == self.mask.size()[0]
        ), "vis and dataset mask do not have the same number of channels."

        # As of Pytorch 1.7.0, complex numbers are partially supported.
        # However, masked_select does not yet work (with gradients)
        # on the complex vis, so hence this awkward step of selecting
        # the reals and imaginaries separately
        re = modelVisibilityCube.real.masked_select(self.mask)
        im = modelVisibilityCube.imag.masked_select(self.mask)

        # we had trouble returning things as re + 1.0j * im,
        # but for some reason torch.complex seems to work OK.
        return torch.complex(re, im)

    @property
    def ground_mask(self) -> torch.Tensor:
        r"""
        The boolean mask, arranged in ground format.

        Returns:
            torch.boolean : 3D mask cube of shape ``(nchan, npix, npix)``

        """
        return utils.packed_cube_to_ground_cube(self.mask)


class Dartboard:
    r"""
    A polar coordinate grid relative to a :class:`~mpol.coordinates.GridCoords` object,
    reminiscent of a dartboard layout. The main utility of this object is to support
    splitting a dataset along radial and azimuthal bins for k-fold cross validation.

    Args:
        coords (GridCoords): an object already instantiated from the GridCoords class.
            If providing this, cannot provide ``cell_size`` or ``npix``.
        q_edges (1D numpy array): an array of radial bin edges to set the dartboard
            cells in :math:`[\mathrm{k}\lambda]`. If ``None``, defaults to 12
            log-linearly radial bins stretching from 0 to the :math:`q_\mathrm{max}`
            represented by ``coords``.
        phi_edges (1D numpy array): an array of azimuthal bin edges to set the
            dartboard cells in [radians], over the domain :math:`[0, \pi]`, which is
            also implicitly mapped to the domain :math:`[-\pi, \pi]` to preserve the
            Hermitian nature of the visibilities. If ``None``, defaults to
            8 equal-spaced azimuthal bins stretched from :math:`0` to :math:`\pi`.
    """

    def __init__(
        self,
        coords: GridCoords,
        q_edges: NDArray[floating[Any]] | None = None,
        phi_edges: NDArray[floating[Any]] | None = None,
    ) -> None:
        self.coords = coords
        self.nchan = 1

        # if phi_edges is not given, we'll instantiate
        if phi_edges is None:
            phi_edges = np.linspace(0, np.pi, num=8 + 1)  # [radians]
        elif not all(0 <= edge <= np.pi for edge in phi_edges):
            raise ValueError("Elements of phi_edges must be between 0 and pi.")

        if q_edges is None:
            # set q edges approximately following inspiration from Petry et al. scheme:
            # https://ui.adsabs.harvard.edu/abs/2020SPIE11449E..1DP/abstract
            # first two bins set to 7m width
            # after third bin, bin width increases linearly until it is
            # 700m at 16km baseline.
            # From 16m to 16km, bin width goes from 7m to 700m.
            # ---
            # We aren't doing *quite* the same thing,
            # just logspacing with a few linear cells at the start.
            q_edges = utils.loglinspace(0, self.q_max, N_log=8, M_linear=5)

        self.q_edges = q_edges
        self.phi_edges = phi_edges

    @property
    def cartesian_qs(self) -> NDArray[floating[Any]]:
        return self.coords.packed_q_centers_2D

    @property
    def cartesian_phis(self) -> NDArray[floating[Any]]:
        return self.coords.packed_phi_centers_2D

    @property
    def q_max(self) -> float:
        return self.coords.q_max

    def get_polar_histogram(
        self, qs: NDArray[floating[Any]], phis: NDArray[floating[Any]]
    ) -> NDArray[floating[Any]]:
        r"""
        Calculate a histogram in polar coordinates, using the bin edges defined by
        ``q_edges`` and ``phi_edges`` during initialization.
        Data coordinates should include the points for the Hermitian visibilities.

        Args:
            qs: 1d array of q values :math:`[\lambda]`
            phis: 1d array of datapoint azimuth values [radians] (must be the same
                length as qs)

        Returns:
            2d integer numpy array of cell counts, i.e., how many datapoints fell into
            each dartboard cell.
        """

        histogram: NDArray
        # make a polar histogram
        histogram, *_ = np.histogram2d( # type:ignore
            qs, phis, bins=[self.q_edges.tolist(), self.phi_edges.tolist()] # type:ignore
        )

        return histogram

    def get_nonzero_cell_indices(
        self, qs: NDArray[floating[Any]], phis: NDArray[floating[Any]]
    ) -> NDArray[integer[Any]]:
        r"""
        Return a list of the cell indices that contain data points, using the bin edges
        defined by ``q_edges`` and ``phi_edges`` during initialization.
        Data coordinates should include the points for the Hermitian visibilities.

        Args:
            qs: 1d array of q values :math:`[\lambda]`
            phis: 1d array of datapoint azimuth values [radians] (must be the same
                length as qs)

        Returns:
            list of cell indices where cell contains at least one datapoint.
        """

        # make a polar histogram
        histogram = self.get_polar_histogram(qs, phis)

        indices = np.argwhere(histogram > 0)  # [i,j] indexes to go to q, phi

        return indices

    def build_grid_mask_from_cells(
        self, cell_index_list: NDArray[integer[Any]]
    ) -> NDArray[np.bool_]:
        r"""
        Create a boolean mask of size ``(npix, npix)`` (in packed format) corresponding
        to the ``vis_gridded`` and ``weight_gridded`` quantities of the
        :class:`~mpol.datasets.GriddedDataset` .

        Args:
            cell_index_list (list): list or iterable containing [q_cell, phi_cell] index
                pairs to include in the mask.

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
