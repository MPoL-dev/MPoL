import numpy as np
from numpy.fft import ifft2, fftfreq, fftshift, ifftshift, rfftfreq
from .constants import arcsec


class Gridder:
    def __init__(self, cell_size, npix, uu, vv, weight, data_re, data_im):
        """
        The Gridder object serves several functions. It uses desired image dimensions (via the ``cell_size`` and ``npix`` arguments) to define a corresponding Fourier plane grid. It also takes in *ungridded* visibility data and stores it to the object.

        Then, the user can decide how to 'grid', or average, the loose visibilities to a more compact representation on the Fourier grid using the `self.grid_visibilities` routine.

        If your goal is to use these gridded visibilities in Regularized Maximum Likelihood imaging, you can export them to the appropriate PyTorch object using the `self.to_pytorch_dataset` routine.

        If you just want to take a quick look at the rough image plane representation of the visibilities, you can view the 'dirty image' and the point spread function or 'dirty beam'. After the visibilities have been gridded, these are available via the `self.dirty_image` and `self.dirty_beam` attributes.
        
        Like the ImageCube class, the Gridder assumes that you are operating with a multi-channel set of visibilities. These routines will still work with single-channel 'continuum' visibilities, they will just have nchan = 1 in the first dimension of most products.

        Args:
            cell_size (float): width of a single square pixel in [arcsec]
            npix (int): number of pixels in the width of the image 
            uu (2d numpy array): (nchan, nvis) length array of u spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
            vv (2d numpy array): (nchan, nvis) length array of v spatial frequency coordinates. Units of [:math:`\mathrm{k}\lambda`]
            weight (2d numpy array): (nchan, nvis) length array of thermal weights. Units of [:math:`1/\mathrm{Jy}^2`]
            data_re (2d numpy array): (nchan, nvis) length array of the real part of the visibility measurements. Units of [:math:`\mathrm{Jy}`]
            data_im (2d numpy array): (nchan, nvis) length array of the imaginary part of the visibility measurements. Units of [:math:`\mathrm{Jy}`]
        """
        # set up the bin edges, centers, etc.
        assert npix % 2 == 0, "Image must have an even number of pixels"

        self.cell_size = cell_size  # arcsec
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
        self.uu = np.concatenate([uu, -uu], axis=1)
        self.vv = np.concatenate([vv, -vv], axis=1)
        self.weight = np.concatenate([weight, weight], axis=1)
        self.data_re = np.concatenate([data_re, data_re], axis=1)
        self.data_im = np.concatenate(
            [data_im, -data_im], axis=1
        )  # the complex conjugates

        # make sure the data visibilities fit within the Fourier grid defined by cell_size and npix.
        assert (
            np.min(self.uu) > u_bin_min
        ), "uu data visibilities outside (more negative than) uu grid bounds. Adjust cell_size and npix."
        assert (
            np.min(self.vv) > v_bin_min
        ), "vv data visibilities outside (more negative than) vv grid bounds. Adjust cell_size and npix."
        assert (
            np.max(self.uu) < u_bin_max
        ), "uu data visibilities outside (greater than) uu grid bounds. Adjust cell_size and npix."
        assert (
            np.max(self.vv) < v_bin_max
        ), "vv data visibilities outside (greater than) vv grid bounds. Adjust cell_size and npix."

        self.nchan = len(self.uu)

        # figure out which cell each visibility lands in, so that
        # we can later assign it the appropriate robust weight for that cell
        # do this by calculating the nearest cell index [0, N] for all samples
        self.index_u = np.array(
            [np.digitize(u_chan, self.u_edges) - 1 for u_chan in self.uu]
        )
        self.index_v = np.array(
            [np.digitize(v_chan, self.v_edges) - 1 for v_chan in self.vv]
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

