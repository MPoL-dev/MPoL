from __future__ import annotations
from functools import cached_property

from typing import Any

import numpy as np
import numpy.fft as np_fft
from numpy import floating
from numpy.typing import NDArray, ArrayLike

import mpol.constants as const
from mpol.exceptions import CellSizeError
from mpol.utils import get_max_spatial_freq, get_maximum_cell_size


class GridCoords:
    r"""
    The GridCoords object uses desired image dimensions (via the ``cell_size`` and ``npix`` arguments) to define a corresponding Fourier plane grid.

    Args:
        cell_size (float): width of a single square pixel in [arcsec]
        npix (int): number of pixels in the width of the image

    The Fourier grid is defined over the domain :math:`[-u,+u]`, :math:`[-v,+v]`, even though for real images, technically we could use an RFFT grid from :math:`[0,+u]` to :math:`[-v,+v]`. The reason we opt for a full FFT grid in this instance is implementation simplicity.

    Images (and their corresponding Fourier transform quantities) are represented as two-dimensional arrays packed as ``[y, x]`` and ``[v, u]``.  This means that an image with dimensions ``(npix, npix)`` will also have a corresponding FFT Fourier grid with shape ``(npix, npix)``. The native :class:`~mpol.gridding.GridCoords` representation assumes the Fourier grid (and thus image) are laid out as one might normally expect an image (i.e., no ``np.fft.fftshift`` has been applied).

    After the object is initialized, instance variables can be accessed, for example

    >>> myCoords = GridCoords(cell_size=0.005, 512)
    >>> myCoords.img_ext

    :ivar dl: image-plane cell spacing in RA direction (assumed to be positive) [radians]
    :ivar dm: image-plane cell spacing in DEC direction [radians]
    :ivar img_ext: The length-4 list of (left, right, bottom, top) expected by routines like ``matplotlib.pyplot.imshow`` in the ``extent`` parameter assuming ``origin='lower'``. Units of [arcsec]
    :ivar packed_x_centers_2D: 2D array of l increasing, with fftshifted applied [arcseconds]. Useful for directly evaluating some function to create a packed cube.
    :ivar packed_y_centers_2D: 2D array of m increasing, with fftshifted applied [arcseconds]. Useful for directly evaluating some function to create a packed cube.
    :ivar sky_x_centers_2D: 2D array of l arranged for evaluating a sky image [arcseconds]. l coordinate increases to the left (as on sky).
    :ivar sky_y_centers_2D: 2D array of m arranged for evaluating a sky image [arcseconds].
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
        self._dimage = self.cell_size * const.arcsec  # [radians]
        self._image_centers = self._dimage * np.arange(npix) - npix // 2

        # Frequency related
        # These properties are identical for both u & v and defined here
        self._df = 1 / (npix * self._dimage) * 1e-3  # [kλ]
        self._f_edges = self._dimage * (np.arange(npix + 1) - npix // 2 - 0.5)  # [kλ]
        self._f_centers = self._df * (np.arange(npix) - npix // 2)
        self._min_f = float(self._f_edges.min())
        self._max_f = float(self._f_edges.max())

        # max u or v freq supported by current grid
        self.max_grid = get_max_spatial_freq(cell_size, npix)

    def __repr__(self):
        return f"GridCoords(cell_size={self.cell_size:.2e}, npix={self.npix})"

    @property
    def cell_size(self) -> float:
        return self._cell_size

    @property
    def npix(self) -> int:
        return self._npix

    @property
    def dl(self) -> float:
        return self._dimage  # [radians]

    @property
    def dm(self) -> float:
        return self._dimage  # [radians]

    @property
    def l_centers(self) -> NDArray[floating[Any]]:
        return self._image_centers

    @property
    def m_centers(self) -> NDArray[floating[Any]]:
        return self._image_centers

    @property
    def ncell_u(self) -> int:
        return self.npix

    @property
    def ncell_v(self) -> int:
        return self.npix

    @property
    def du(self) -> float:
        return self._df

    @property
    def dv(self) -> float:
        return self._df

    @property
    def u_edges(self) -> NDArray[floating[Any]]:
        return self._f_edges

    @property
    def v_edges(self) -> NDArray[floating[Any]]:
        return self._f_edges

    @property
    def u_centers(self) -> NDArray[floating[Any]]:
        return self._f_centers

    @property
    def v_centers(self) -> NDArray[floating[Any]]:
        return self._f_centers

    @property
    def u_bin_min(self) -> float:
        return self._min_f

    @property
    def v_bin_min(self) -> float:
        return self._min_f

    @property
    def u_bin_max(self) -> float:
        return self._max_f

    @property
    def v_bin_max(self) -> float:
        return self._max_f

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
        ]  # [kλ]

    # --------------------------------------------------------------------------
    # Non-identical u & v properties
    # --------------------------------------------------------------------------
    @cached_property
    def sky_u_centers_2D(self) -> NDArray[floating[Any]]:
        # only useful for plotting a sky_vis
        # uu increasing, no fftshift
        return np.tile(self.u_centers, (self.ncell_u, 1))

    @cached_property
    def sky_v_centers_2D(self) -> NDArray[floating[Any]]:
        # only useful for plotting a sky_vis
        # vv increasing, no fftshift
        return np.tile(self.v_centers, (self.ncell_v, 1)).T

    @cached_property
    def packed_u_centers_2D(self) -> NDArray[floating[Any]]:
        # for evaluating a packed vis
        # uu increasing, fftshifted
        return np_fft.fftshift(self.sky_u_centers_2D)

    @cached_property
    def packed_v_centers_2D(self) -> NDArray[floating[Any]]:
        # for evaluating a packed vis
        # vv increasing + fftshifted
        return np_fft.fftshift(self.sky_v_centers_2D)

    @cached_property
    def sky_q_centers_2D(self) -> NDArray[floating[Any]]:
        return np.sqrt(self.sky_u_centers_2D**2 + self.sky_v_centers_2D**2)  # [kλ]

    @cached_property
    def sky_phi_centers_2D(self) -> NDArray[floating[Any]]:
        # https://en.wikipedia.org/wiki/Atan2
        return np.arctan2(self.sky_v_centers_2D, self.sky_u_centers_2D)  # (pi, pi]

    @cached_property
    def packed_q_centers_2D(self) -> NDArray[floating[Any]]:
        # for evaluating a packed vis in polar coordinates
        # q increasing, fftshifted
        return np_fft.fftshift(self.sky_q_centers_2D)

    @cached_property
    def packed_phi_centers_2D(self) -> NDArray[floating[Any]]:
        # for evaluating a packed vis in polar coordinates
        # phi increasing, fftshifted
        return np_fft.fftshift(self.sky_phi_centers_2D)

    @cached_property
    def q_max(self) -> float:
        # outer edge [klambda]
        return float(np.abs(self.packed_q_centers_2D).max() + np.sqrt(2) * self.du)

    @cached_property
    def x_centers_2D(self) -> NDArray[floating[Any]]:
        return np.tile(self.l_centers / const.arcsec, (self.npix, 1))  # [arcsec]

    @cached_property
    def y_centers_2D(self) -> NDArray[floating[Any]]:
        return np.tile(self.m_centers / const.arcsec, (self.npix, 1)).T  # [arcsec]

    @cached_property
    def packed_x_centers_2D(self) -> NDArray[floating[Any]]:
        return np.fft.fftshift(self.x_centers_2D)  # [arcsec]

    @cached_property
    def packed_y_centers_2D(self) -> NDArray[floating[Any]]:
        return np.fft.fftshift(self.y_centers_2D)  # [arcsec]

    @property
    def sky_x_centers_2D(self) -> NDArray[floating[Any]]:
        # for evaluating a sky image
        # ll mirrored, increasing, no fftshift
        return np.fliplr(self.x_centers_2D)  # [arcsec]

    @property
    def sky_y_centers_2D(self) -> NDArray[floating[Any]]:
        # for evaluating a sky image
        # mm increasing, no fftshift
        return self.y_centers_2D  # [arcsec]

    def check_data_fit(self, uu: ArrayLike, vv: ArrayLike) -> None:
        r"""
        Test whether loose data visibilities fit within the Fourier grid defined by cell_size and npix.

        Args:
            uu (np.array): array of u spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
            vv (np.array): array of v spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]

        Returns:
            ``True`` if all visibilities fit within the Fourier grid defined by ``[u_bin_min, u_bin_max, v_bin_min, v_bin_max]``. Otherwise an ``AssertionError`` is raised on the first violated boundary.
        """

        # max freq in dataset
        max_uu_vv = np.abs(np.concatenate([uu, vv])).max()

        # max freq needed to support dataset
        max_cell_size = get_maximum_cell_size(max_uu_vv)

        if np.abs(uu).max() > self.max_grid:
            raise CellSizeError(
                "Dataset contains uu spatial frequency measurements larger "
                "than those in the proposed model image. "
                f"Decrease cell_size below {max_cell_size} arcsec."
            )

        if np.abs(vv).max() > self.max_grid:
            raise CellSizeError(
                "Dataset contains vv spatial frequency measurements larger "
                "than those in the proposed model image. "
                f"Decrease cell_size below {max_cell_size} arcsec."
            )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GridCoords):
            # don't attempt to compare against different types
            return NotImplemented

        # GridCoords objects are considered equal if they have the same cell_size and npix, since
        # all other attributes are derived from these two core properties.
        return bool(self.cell_size == other.cell_size and self.npix == other.npix)
