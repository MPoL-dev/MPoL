from __future__ import annotations

from functools import cached_property
from typing import Any

import numpy as np
import numpy.fft as np_fft
import numpy.typing as npt
import torch

import mpol.constants as const
from mpol.exceptions import CellSizeError
from mpol.utils import get_maximum_cell_size


class GridCoords:
    r"""
    The GridCoords object uses desired image dimensions (via the ``cell_size`` and
    ``npix`` arguments) to define a corresponding Fourier plane grid.

    Parameters
    ----------
    cell_size : float
        width of a single square pixel in [arcsec]
    npix : int
        number of pixels in the width of the image

    The Fourier grid is defined over the domain :math:`[-u,+u]`, :math:`[-v,+v]`, even
    though for real images, technically we could use an RFFT grid from :math:`[0,+u]` to
    :math:`[-v,+v]`. The reason we opt for a full FFT grid in this instance is
    implementation simplicity.

    Images (and their corresponding Fourier transform quantities) are represented as
    two-dimensional arrays packed as ``[y, x]`` and ``[v, u]``.  This means that an
    image with dimensions ``(npix, npix)`` will also have a corresponding FFT Fourier
    grid with shape ``(npix, npix)``. The native :class:`~mpol.gridding.GridCoords`
    representation assumes the Fourier grid (and thus image) are laid out as one might
    normally expect an image (i.e., no ``np.fft.fftshift`` has been applied).

    After the object is initialized, instance variables can be accessed, for example

    >>> myCoords = GridCoords(cell_size=0.005, npix=512)
    >>> myCoords.img_ext

    :ivar dl: image-plane cell spacing in RA direction (assumed to be positive)
        [radians]
    :ivar dm: image-plane cell spacing in DEC direction [radians]
    :ivar img_ext: The length-4 list of (left, right, bottom, top) expected by routines
        like ``matplotlib.pyplot.imshow`` in the ``extent`` parameter assuming
        ``origin='lower'``. Units of [arcsec]
    :ivar packed_x_centers_2D: 2D array of l increasing, with fftshifted applied
        [arcseconds]. Useful for directly evaluating some function to create a
        packed cube.
    :ivar packed_y_centers_2D: 2D array of m increasing, with fftshifted applied
        [arcseconds]. Useful for directly evaluating some function to create a
        packed cube.
    :ivar sky_x_centers_2D: 2D array of l arranged for evaluating a sky image
        [arcseconds]. l coordinate increases to the left (as on sky).
    :ivar sky_y_centers_2D: 2D array of m arranged for evaluating a sky image
        [arcseconds].
    :ivar du: Fourier-plane cell spacing in East-West direction
        [:math:`\lambda`]
    :ivar dv: Fourier-plane cell spacing in North-South direction
        [:math:`\lambda`]
    :ivar u_centers: 1D array of cell centers in East-West direction
        [:math:`\lambda`].
    :ivar v_centers: 1D array of cell centers in North-West direction
        [:math:`\lambda`].
    :ivar u_edges: 1D array of cell edges in East-West direction
        [:math:`\lambda`].
    :ivar v_edges: 1D array of cell edges in North-South direction
        [:math:`\lambda`].
    :ivar u_bin_min: minimum u edge [:math:`\lambda`]
    :ivar u_bin_max: maximum u edge [:math:`\lambda`]
    :ivar v_bin_min: minimum v edge [:math:`\lambda`]
    :ivar v_bin_max: maximum v edge [:math:`\lambda`]
    :ivar max_grid: maximum spatial frequency enclosed by Fourier grid
        [:math:`\lambda`]
    :ivar vis_ext: length-4 list of (left, right, bottom, top) expected by routines
        like ``matplotlib.pyplot.imshow`` in the ``extent`` parameter assuming
        ``origin='lower'``. Units of [:math:`\lambda`]
    :ivar vis_ext_Mlam: like vis_ext, but in units of [:math:`\mathrm{M}\lambda`].
    """

    def __init__(self, cell_size: float, npix: int):
        if npix <= 0 or not (npix % 2 == 0):
            raise ValueError("Image must have a positive and even number of pixels.")

        if cell_size <= 0:
            raise ValueError("cell_size must be a positive real number.")

        # Imply to users that GridCoords instance is read-only and new instance
        # is the approach if changing values
        self._cell_size = cell_size
        self._npix = npix

        # Image related
        self._image_pixel_width = cell_size * const.arcsec  # [radians]
        self._image_centers = self._image_pixel_width * (
            np.arange(npix) - npix // 2
        )  # [radians]

        # Spatial frequency related
        # These properties are identical for both u & v and defined here
        # edges and centers return fftspaced arrays (not packed, though)
        # All units in [λ]
        self._uv_pixel_width = 1 / (npix * self._image_pixel_width)
        self.uv_edges = self._uv_pixel_width * (np.arange(npix + 1) - npix // 2 - 0.5)
        self._uv_centers = self._uv_pixel_width * (np.arange(npix) - npix // 2)
        self._min_uv = float(self.uv_edges.min())
        self._max_uv = float(self.uv_edges.max())

    def __repr__(self):
        return f"GridCoords(cell_size={self.cell_size:.2e}, npix={self.npix})"
        # the output spatial frequencies of the FFT routine

    @property
    def cell_size(self) -> float:
        return self._cell_size

    @property
    def npix(self) -> int:
        return self._npix

    @property
    def dl(self) -> float:
        return self._image_pixel_width  # [radians]

    @property
    def dm(self) -> float:
        return self._image_pixel_width  # [radians]

    @property
    def l_centers(self) -> npt.NDArray[np.floating[Any]]:
        return self._image_centers

    @property
    def m_centers(self) -> npt.NDArray[np.floating[Any]]:
        return self._image_centers

    @property
    def npix_u(self) -> int:
        return self.npix

    @property
    def npix_v(self) -> int:
        return self.npix

    @property
    def du(self) -> float:
        return self._uv_pixel_width

    @property
    def dv(self) -> float:
        return self._uv_pixel_width

    @property
    def u_edges(self) -> npt.NDArray[np.floating[Any]]:
        return self.uv_edges

    @property
    def v_edges(self) -> npt.NDArray[np.floating[Any]]:
        return self.uv_edges

    @property
    def u_centers(self) -> npt.NDArray[np.floating[Any]]:
        return self._uv_centers

    @property
    def v_centers(self) -> npt.NDArray[np.floating[Any]]:
        return self._uv_centers

    @property
    def u_bin_min(self) -> float:
        return self._min_uv

    @property
    def v_bin_min(self) -> float:
        return self._min_uv

    @property
    def u_bin_max(self) -> float:
        return self._max_uv

    @property
    def v_bin_max(self) -> float:
        return self._max_uv

    @property
    def max_uv_grid_value(self) -> float:
        return self._max_uv

    @property
    def img_ext(self) -> list[float]:
        # calculate the image extent
        # say we had 10 pixels representing centers -5, -4, -3, ...
        # it should go from -5.5 to +4.5
        lmax = self.cell_size * (self.npix // 2 - 0.5)
        lmin = -self.cell_size * (self.npix // 2 + 0.5)
        return [lmax, lmin, lmin, lmax]  # arcsecs

    @property
    def vis_ext(self) -> list[float]:
        return [
            self.u_bin_min,
            self.u_bin_max,
            self.v_bin_min,
            self.v_bin_max,
        ]  # [λ]

    @property
    def vis_ext_Mlam(self) -> list[float]:
        return [1e-6 * edge for edge in self.vis_ext]

    @cached_property
    def ground_u_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        # only useful for plotting a sky_vis
        # uu increasing, no fftshift
        # tile replicates the 1D u_centers array to a 2D array the size of the full
        # UV grid
        return np.tile(self.u_centers, (self.npix_u, 1))

    @cached_property
    def ground_v_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        # only useful for plotting a sky_vis
        # vv increasing, no fftshift
        return np.tile(self.v_centers, (self.npix_v, 1)).T

    @cached_property
    def packed_u_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        # for evaluating a packed vis
        # uu increasing, fftshifted
        return np_fft.fftshift(self.ground_u_centers_2D)

    @cached_property
    def packed_v_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        # for evaluating a packed vis
        # vv increasing + fftshifted
        return np_fft.fftshift(self.ground_v_centers_2D)

    @cached_property
    def ground_q_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        return np.sqrt(
            self.ground_u_centers_2D**2 + self.ground_v_centers_2D**2
        )  # [kλ]

    @cached_property
    def sky_phi_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        # https://en.wikipedia.org/wiki/Atan2
        return np.arctan2(
            self.ground_v_centers_2D, self.ground_u_centers_2D
        )  # (pi, pi]

    @cached_property
    def packed_q_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        # for evaluating a packed vis in polar coordinates
        # q increasing, fftshifted
        return np_fft.fftshift(self.ground_q_centers_2D)

    @cached_property
    def packed_phi_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        # for evaluating a packed vis in polar coordinates
        # phi increasing, fftshifted
        return np_fft.fftshift(self.sky_phi_centers_2D)

    @cached_property
    def q_max(self) -> float:
        # outer edge [lambda]
        return float(np.abs(self.packed_q_centers_2D).max() + np.sqrt(2) * self.du)

    @cached_property
    def x_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        return np.tile(self.l_centers / const.arcsec, (self.npix, 1))  # [arcsec]

    @cached_property
    def y_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        return np.tile(self.m_centers / const.arcsec, (self.npix, 1)).T  # [arcsec]

    @cached_property
    def packed_x_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        return np.fft.fftshift(self.x_centers_2D)  # [arcsec]

    @cached_property
    def packed_y_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        return np.fft.fftshift(self.y_centers_2D)  # [arcsec]

    @property
    def sky_x_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        # for evaluating a sky image
        # ll mirrored, increasing, no fftshift
        return np.fliplr(self.x_centers_2D)  # [arcsec]

    @property
    def sky_y_centers_2D(self) -> npt.NDArray[np.floating[Any]]:
        # for evaluating a sky image
        # mm increasing, no fftshift
        return self.y_centers_2D  # [arcsec]

    def check_data_fit(
        self,
        uu: torch.Tensor | npt.NDArray[np.floating[Any]],
        vv: torch.Tensor | npt.NDArray[np.floating[Any]],
    ) -> bool:
        r"""
        Test whether loose data visibilities fit within the Fourier grid defined by
        cell_size and npix.

        Parameters
        ----------
        uu : :class:`torch.Tensor`
            u spatial frequency coordinates.
            Units of [:math:`\lambda`]
        vv : :class:`torch.Tensor`
            v spatial frequency coordinates.
            Units of [:math:`\lambda`]

        Returns
        -------
        bool
            ``True`` if all visibilities fit within the Fourier grid defined by
            ``[u_bin_min, u_bin_max, v_bin_min, v_bin_max]``. Otherwise a
            :class:`mpol.exceptions.CellSizeError` is raised on the first violated
            boundary.
        """

        # we need this routine to work with both numpy.ndarray or torch.Tensor
        # because it is called for DirtyImager setup (numpy only)
        # so we'll cast to tensor as a precaution
        uu = torch.as_tensor(uu)
        vv = torch.as_tensor(vv)

        # max freq in dataset
        max_uu_vv = np.abs(np.concatenate([uu, vv])).max()
        max_uu_vv = torch.max(torch.abs(torch.concatenate([uu, vv]))).item()

        # max freq needed to support dataset
        max_cell_size = get_maximum_cell_size(max_uu_vv)

        if torch.max(torch.abs(uu)) > self.max_uv_grid_value:
            raise CellSizeError(
                "Dataset contains uu spatial frequency measurements larger "
                "than those in the proposed model image. "
                f"Decrease cell_size below {max_cell_size} arcsec."
            )

        if torch.max(torch.abs(vv)) > self.max_uv_grid_value:
            raise CellSizeError(
                "Dataset contains vv spatial frequency measurements larger "
                "than those in the proposed model image. "
                f"Decrease cell_size below {max_cell_size} arcsec."
            )

        return True

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GridCoords):
            # don't attempt to compare against different types
            return NotImplemented

        # GridCoords objects are considered equal if they have the same cell_size and
        # npix, since all other attributes are derived from these two core properties.
        return bool(self.cell_size == other.cell_size and self.npix == other.npix)
