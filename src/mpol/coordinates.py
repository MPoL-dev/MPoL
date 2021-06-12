import numpy as np
from numpy.fft import fftfreq, fftshift, ifft2, ifftshift, rfftfreq

from .constants import arcsec
from .utils import get_max_spatial_freq, get_maximum_cell_size


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
        self.ncell_u = self.npix
        self.ncell_v = self.npix

        # calculate the image extent
        # say we had 10 pixels representing centers -5, -4, -3, ...
        # it should go from -5.5 to +4.5
        lmax = cell_size * (self.npix // 2 - 0.5)
        lmin = -cell_size * (self.npix // 2 + 0.5)
        self.img_ext = [lmax, lmin, lmin, lmax]  # arcsecs

        self.dl = cell_size * arcsec  # [radians]
        self.dm = cell_size * arcsec  # [radians]

        int_l_centers = np.arange(self.npix) - self.npix // 2
        int_m_centers = np.arange(self.npix) - self.npix // 2
        self.l_centers = self.dl * int_l_centers  # [radians]
        self.m_centers = self.dm * int_m_centers  # [radians]

        # the output spatial frequencies of the FFT routine
        self.du = 1 / (self.npix * self.dl) * 1e-3  # [kλ]
        self.dv = 1 / (self.npix * self.dm) * 1e-3  # [kλ]

        # define the max/min of the FFT grid
        # because we store images as [y, x]
        # this means we store visibilities as [v, u]
        int_u_edges = np.arange(self.ncell_u + 1) - self.ncell_v // 2 - 0.5
        int_v_edges = np.arange(self.ncell_v + 1) - self.ncell_v // 2 - 0.5

        self.u_edges = self.du * int_u_edges  # [kλ]
        self.v_edges = self.dv * int_v_edges  # [kλ]

        int_u_centers = np.arange(self.ncell_u) - self.ncell_u // 2
        int_v_centers = np.arange(self.ncell_v) - self.ncell_v // 2
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

        # max u or v freq supported by current grid
        self.max_grid = get_max_spatial_freq(self.cell_size, self.npix)

        # only useful for plotting a sky_vis... uu, vv increasing, no fftshift
        self.sky_u_centers_2D, self.sky_v_centers_2D = np.meshgrid(
            self.u_centers, self.v_centers, indexing="xy"
        )  # cartesian indexing (default)

        # only useful for plotting... uu, vv increasing, no fftshift
        self.sky_q_centers_2D = np.sqrt(
            self.sky_u_centers_2D ** 2 + self.sky_v_centers_2D ** 2
        )  # [kλ]

        # https://en.wikipedia.org/wiki/Atan2
        self.sky_phi_centers_2D = np.arctan2(
            self.sky_v_centers_2D, self.sky_u_centers_2D
        )  # (pi, pi]

        # for evaluating a packed vis... uu, vv increasing + fftshifted
        self.packed_u_centers_2D = np.fft.fftshift(self.sky_u_centers_2D)
        self.packed_v_centers_2D = np.fft.fftshift(self.sky_v_centers_2D)

        # and in polar coordinates too
        self.packed_q_centers_2D = np.fft.fftshift(self.sky_q_centers_2D)
        self.packed_phi_centers_2D = np.fft.fftshift(self.sky_phi_centers_2D)

        self.q_max = (
            np.max(np.abs(self.packed_q_centers_2D)) + np.sqrt(2) * self.du
        )  # outer edge [klambda]

        # x_centers_2D and y_centers_2D are just l and m in units of arcsec
        x_centers_2D, y_centers_2D = np.meshgrid(
            self.l_centers / arcsec, self.m_centers / arcsec, indexing="xy"
        )  # [arcsec] cartesian indexing (default)

        # for evaluating a packed cube... ll, mm increasing + fftshifted
        self.packed_x_centers_2D = np.fft.fftshift(x_centers_2D)  # [arcsec]
        self.packed_y_centers_2D = np.fft.fftshift(y_centers_2D)  # [arcsec]

        # for evaluating a sky image... ll mirrored, mm increasing, no fftshift
        self.sky_y_centers_2D = y_centers_2D  # [arcsec]
        self.sky_x_centers_2D = np.fliplr(x_centers_2D)  # [arcsec]

    def check_data_fit(self, uu, vv):
        r"""
        Test whether loose data visibilities fit within the Fourier grid defined by cell_size and npix.

        Args:
            uu (np.array): array of u spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
            vv (np.array): array of v spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]

        Returns:
            ``True`` if all visibilities fit within the Fourier grid defined by ``[u_bin_min, u_bin_max, v_bin_min, v_bin_max]``. Otherwise an ``AssertionError`` is raised on the first violated boundary.
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

    def __eq__(self, other):
        if not isinstance(other, GridCoords):
            # don't attempt to compare against different types
            return NotImplemented

        # GridCoords objects are considered equal if they have the same cell_size and npix, since
        # all other attributes are derived from these two core properties.
        return self.cell_size == other.cell_size and self.npix == other.npix


def _setup_coords(self, cell_size=None, npix=None, coords=None, nchan=None):
    r"""
    Convenience helper to setup coordinate objects inside BaseCube and ImageCube classes. This is meant to be called inside ``__init__``, and will create the instance attributes on ``self``.

    Args:
        self: reference to instance object of class.
        cell_size (float): the width of a pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        nchan (int): the number of channels in the image

    Returns: None.
    """
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

    if nchan is not None:
        self.nchan = nchan
    else:
        self.nchan = 1
