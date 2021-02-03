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

    Images (and their corresponding Fourier transform quantities) are represented as two-dimensional arrays packed as ``[y, x]`` and ``[v, u]``.  This means that an image with dimensions ``(npix, npix)`` will have a corresponding RFFT Fourier grid with shape ``(npix, npix//2 + 1)`` because the RFFT is performed over the trailing coordinate, in this case :math:`u`. 

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
        int_u_edges = np.arange(self.npix // 2 + 2) - 0.5
        int_v_edges = np.arange(self.npix + 1) - self.npix // 2 - 0.5
        self.u_edges = self.du * int_u_edges  # [kλ]
        self.v_edges = self.dv * int_v_edges  # [kλ]

        int_u_centers = np.arange(self.npix // 2 + 1)
        int_v_centers = np.arange(self.npix) - self.npix // 2
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
        """
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


class Gridder:
    """
    The Gridder object serves several functions. It uses desired image dimensions (via the ``cell_size`` and ``npix`` arguments) to define a corresponding Fourier plane grid. Because we are dealing with real images, the Fourier grid is defined by the RFFT grid. This means the visibility grid will have shape ``(npix, npix//2 + 1)`` corresponding to the RFFT output of an image with ``cell_size` and dimensions ``(npix, npix)``. A pre-computed ``GridCoords`` can also be supplied in lieu of ``cell_size`` and ``npix.``

    The ``Gridder`` object accepts "loose" *ungridded* visibility data and stores it to the object. The visibility data can be over the full :math:`[-u,u]` and :math:`[-v,v]` domain, the Gridder will mirror the loose data into the RFFT domain (:math:`[0,u]` and :math:`[-v,v]`) automatically.

    Then, the user can decide how to 'grid', or average, the loose visibilities to a more compact representation on the Fourier grid using the ``self.grid_visibilities`` routine.

    If your goal is to use these gridded visibilities in Regularized Maximum Likelihood imaging, you can export them to the appropriate PyTorch object using the `self.to_pytorch_dataset` routine.

    If you just want to take a quick look at the rough image plane representation of the visibilities, you can view the 'dirty image' and the point spread function or 'dirty beam'. After the visibilities have been gridded, these are available via the `self.dirty_image` and `self.dirty_beam` attributes.

    Like the ImageCube class, the Gridder assumes that you are operating with a multi-channel set of visibilities. These routines will still work with single-channel 'continuum' visibilities, they will just have nchan = 1 in the first dimension of most products.

    Args:
        cell_size (float): width of a single square pixel in [arcsec]
        npix (int): number of pixels in the width of the image
        gridCoords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
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
        gridCoords=None,
        uu=None,
        vv=None,
        weight=None,
        data_re=None,
        data_im=None,
    ):

        if gridCoords:
            assert (
                npix is None and cell_size is None
            ), "npix and cell_size must be empty if precomputed GridCoords are supplied."
            self.gridCoords = gridCoords

        elif npix or cell_size:
            assert (
                gridCoords is None
            ), "GridCoords must be empty if npix and cell_size are supplied."

            self.gridCoords = GridCoords(cell_size=cell_size, npix=npix)

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

        # within each channel, expand and overwrite the vectors to include complex conjugates
        uu_full = np.concatenate([uu, -uu], axis=1)
        vv_full = np.concatenate([vv, -vv], axis=1)
        weight_full = np.concatenate([weight, weight], axis=1)
        data_re_full = np.concatenate([data_re, data_re], axis=1)
        data_im_full = np.concatenate(
            [data_im, -data_im], axis=1
        )  # the complex conjugates

        # The RFFT outputs u in the range [0, +] and v in the range [-, +],
        # but the dataset contains measurements at u [-,+] and v [-, +].
        # Find all the u < 0 points and convert them via complex conj
        ind_u_neg = uu_full < 0.0
        uu_full[ind_u_neg] *= -1.0  # swap axes so all u > 0
        vv_full[ind_u_neg] *= -1.0  # swap axes
        data_im_full[ind_u_neg] *= -1.0  # complex conjugate

        self.gridCoords.check_data_fit(uu=uu_full, vv=vv_full)

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
            [np.digitize(u_chan, self.gridCoords.u_edges) - 1 for u_chan in self.uu]
        )
        self.index_v = np.array(
            [np.digitize(v_chan, self.gridCoords.v_edges) - 1 for v_chan in self.vv]
        )

    def grid_visibilities(self, weighting="uniform", robust=None, taper_function=None):
        """
        Grid the loose data visibilities to the Fourier grid defined by `cell_size` and `npix`.

        Specify weighting of "natural", "uniform", or "briggs", following CASA tclean. If "briggs", specify robust in [-2, 2].
        If specifying a taper function, then it is assumed to be f(u, v).
        """

        if taper_function is None:
            tapering_weights = np.ones_like(self.weight)
        else:
            tapering_weights = taper_function(self.uu, self.vv)

        # create the cells as edges around the existing points
        # note that at this stage, the bins are strictly increasing
        # when in fact, later on, we'll need to put this into fftshift format for the FFT
        cell_weight = np.empty((self.nchan, self.npix, self.npix))
        cell_weight, junk, junk = np.histogram2d(
            self.vv, self.uu, bins=[self.v_edges, self.u_edges], weights=self.weight,
        )

        # calculate the density weights
        # the density weights have the same shape as the re, im samples.
        if weighting == "natural":
            density_weights = np.ones_like(self.weight)
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
                np.sum(cell_weight ** 2) / np.sum(self.weight)
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

        self.C = 1 / np.sum(tapering_weights * density_weights * self.weight)

        # grid the reals and imaginaries
        VV_g_real, junk, junk = np.histogram2d(
            self.vv,
            self.uu,
            bins=[self.v_edges, self.u_edges],
            weights=self.data_re * tapering_weights * density_weights * self.weight,
        )

        VV_g_imag, junk, junk = np.histogram2d(
            self.vv,
            self.uu,
            bins=[self.v_edges, self.u_edges],
            weights=self.data_im * tapering_weights * density_weights * self.weight,
        )

        self.VV_g = VV_g_real + VV_g_imag * 1.0j

        # do the beam too
        beam_V_real, junk, junk = np.histogram2d(
            self.vv,
            self.uu,
            bins=[self.v_edges, self.u_edges],
            weights=tapering_weights * density_weights * self.weight,
        )
        self.beam_V = beam_V_real

        # instantiate uncertainties for each averaged visibility.
        # self.VV_uncertainty = values
        # self.VV_uncertainty = None, for routines which it's not implemented yet.

    @property
    def dirty_beam(self):
        """
        Compute the dirty beam corresponding to the gridded visibilities.

        Returns: numpy image cube with a dirty beam for each channel.
        """
        # if we're sticking to the dirty beam and image equations in Briggs' Ph.D. thesis,
        # no correction for du or dv prefactors needed here
        # that is because we are using the FFT to compute an already discretized equation, not
        # approximating a continuous equation.

        self.beam = np.fliplr(
            np.fft.fftshift(
                self.npix ** 2
                * np.fft.ifftn(np.fft.fftshift(self.C * self.beam_V), axes=(0, 1))
            )
        ).real  # Jy/radians^2

        return self.beam

    def get_dirty_image(self, unit="Jy/beam"):
        """
        Calculate the dirty image.

        Args:
            unit (string): what unit should the image be in. Default is "Jy/beam". If "Jy/arcsec^2", then the effective area of the dirty beam will be used to convert from "Jy/beam" to "Jy/arcsec^2".

        Returns: numpy image cube with a dirty image for each channel.
        """

        if unit not in ["Jy/beam", "Jy/arcsec^2"]:
            raise ValueError("Unknown unit", unit)

        img = np.fliplr(
            np.fft.fftshift(
                self.npix ** 2
                * np.fft.ifftn(np.fft.fftshift(self.C * self.VV_g), axes=(0, 1))
            )
        ).real  # Jy/radians^2

        if unit == "Jy/beam":
            return img
        else:
            pass
            # return img / beam_area

