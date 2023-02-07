from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.utils.data as torch_ud
from numpy import floating, integer
from numpy.typing import ArrayLike, NDArray

from mpol.coordinates import GridCoords
from mpol.exceptions import WrongDimensionError

from . import spheroidal_gridding, utils
from .constants import *
from .utils import loglinspace


class GriddedDataset:
    r"""
    Args:
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        vis_gridded (torch complex): the gridded visibility data stored in a "packed" format (pre-shifted for fft)
        weight_gridded (torch double): the weights corresponding to the gridded visibility data, also in a packed format
        mask (torch boolean): a boolean mask to index the non-zero locations of ``vis_gridded`` and ``weight_gridded`` in their packed format.
        nchan (int): the number of channels in the image (default = 1).
        device (torch.device) : the desired device of the dataset. If ``None``, defalts to current device.

    After initialization, the GriddedDataset provides the non-zero cells of the gridded visibilities and weights as a 1D vector via the following instance variables. This means that any individual channel information has been collapsed.

    :ivar vis_indexed: 1D complex tensor of visibility data
    :ivar weight_indexd: 1D tensor of weight values

    If you index the output of the Fourier layer in the same manner using ``self.mask`` (as done internally within :class:`~mpol.connectors.DataConnector`), then the model and data visibilities can be directly compared using a loss function.
    """

    def __init__(
        self,
        *,
        coords: GridCoords,
        vis_gridded: torch.Tensor,
        weight_gridded: torch.Tensor,
        mask: torch.Tensor,
        nchan: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.coords = coords
        self.nchan = nchan

        self.vis_gridded = torch.tensor(vis_gridded, device=device)
        self.weight_gridded = torch.tensor(weight_gridded, device=device)
        self.mask = torch.tensor(mask, device=device)

        # pre-index the values
        # note that these are *collapsed* across all channels
        # 1D array
        self.vis_indexed = self.vis_gridded[self.mask]
        self.weight_indexed = self.weight_gridded[self.mask]

    @classmethod
    def from_image_properties(
        cls,
        cell_size: float,
        npix: int,
        *,
        vis_gridded: torch.Tensor,
        weight_gridded: torch.Tensor,
        mask: torch.Tensor,
        nchan: int = 1,
        device: torch.device = torch.device("cpu"),
    ):
        """Alternative method to instantiate a GriddedDataset object from cell_size and npix.

        Args:
            cell_size (float): the width of a pixel [arcseconds]
            npix (int): the number of pixels per image side
            vis_gridded (torch complex): the gridded visibility data stored in a "packed" format (pre-shifted for fft)
            weight_gridded (torch double): the weights corresponding to the gridded visibility data, also in a packed format
            mask (torch boolean): a boolean mask to index the non-zero locations of ``vis_gridded`` and ``weight_gridded`` in their packed format.
            nchan (int): the number of channels in the image (default = 1).
            device (torch.device) : the desired device of the dataset. If ``None``, defalts to current device.
        """
        return cls(
            coords=GridCoords(cell_size, npix),
            vis_gridded=vis_gridded,
            weight_gridded=weight_gridded,
            mask=mask,
            nchan=nchan,
            device=device,
        )

    def add_mask(
        self, mask: ArrayLike, device: torch.device = torch.device("cpu")
    ) -> None:
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
    def ground_mask(self) -> torch.Tensor:
        r"""
        The boolean mask, arranged in ground format.

        Returns:
            torch.boolean : 3D mask cube of shape ``(nchan, npix, npix)``

        """
        return utils.packed_cube_to_ground_cube(self.mask)

    def to(self, device: torch.device = torch.device("cpu")) -> GriddedDataset:
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
class UVDataset(torch_ud.Dataset):
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
        uu: NDArray[floating[Any]],
        vv: NDArray[floating[Any]],
        weights: NDArray[floating[Any]],
        data_re: NDArray[floating[Any]],
        data_im: NDArray[floating[Any]],
        cell_size: float | None = None,
        npix: int | None = None,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ):
        # ensure that all vectors are the same shape
        if not all(
            array.shape == uu.shape for array in [vv, weights, data_re, data_im]
        ):
            raise WrongDimensionError("All dataset inputs must be the same shape.")

        if uu.ndim == 1:
            uu = np.atleast_2d(uu)
            vv = np.atleast_2d(vv)
            data_re = np.atleast_2d(data_re)
            data_im = np.atleast_2d(data_im)
            weights = np.atleast_2d(weights)

        if np.any(weights <= 0.0):
            raise ValueError("Not all thermal weights are positive, check inputs.")

        self.nchan = uu.shape[0]
        self.gridded = False

        if cell_size is not None and npix is not None:
            (
                uu,
                vv,
                grid_mask,
                weights,
                data_re,
                data_im,
            ) = spheroidal_gridding.grid_dataset(
                uu,
                vv,
                weights,
                data_re,
                data_im,
                cell_size,
                npix,
            )

            # grid_mask (nchan, npix, npix//2 + 1) bool: a boolean array the same size as the output of the RFFT
            # designed to directly index into the output to evaluate against pre-gridded visibilities.
            self.grid_mask = torch.tensor(grid_mask, dtype=torch.bool, device=device)
            self.cell_size = cell_size * arcsec  # [radians]
            self.npix = npix
            self.gridded = True

        self.uu = torch.tensor(uu, dtype=torch.double, device=device)  # klambda
        self.vv = torch.tensor(vv, dtype=torch.double, device=device)  # klambda
        self.weights = torch.tensor(
            weights, dtype=torch.double, device=device
        )  # 1/Jy^2
        self.re = torch.tensor(data_re, dtype=torch.double, device=device)  # Jy
        self.im = torch.tensor(data_im, dtype=torch.double, device=device)  # Jy

        # TODO: store kwargs to do something for antenna self-cal

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            self.uu[index],
            self.vv[index],
            self.weights[index],
            self.re[index],
            self.im[index],
        )

    def __len__(self) -> int:
        return len(self.uu)
