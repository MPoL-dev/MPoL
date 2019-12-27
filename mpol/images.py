import numpy as np
import torch
from torch import nn

from mpol import gridding
from mpol.constants import *
import mpol.utils


class ImageCube(nn.Module):
    """
    Initialize an ImageCube.

    Args:
        npix (int): the number of pixels per image side
        cell_size (float): the size of a pixel in arcseconds
        nchan (int): the number of channels in the image
        velocity_axis (list): vector of velocities (in km/s) corresponding to nchan. Channels should be spaced approximately equidistant in velocity but need not be strictly exact.
        cube (PyTorch tensor w/ `requires_grad = True`): an image cube to initialize the model with. If None, assumes cube is all ones.
    """

    def __init__(
        self,
        npix=None,
        cell_size=None,
        nchan=None,
        velocity_axis=None,
        cube=None,
        **kwargs
    ):

        super().__init__()
        assert npix % 2 == 0, "npix must be even (for now)"
        self.npix = int(npix)

        assert cell_size > 0.0, "cell_size must be positive (arcseconds)"
        self.cell_size = cell_size * arcsec  # [radians]
        # cell_size is also the differential change in sky angles
        # dll = dmm = cell_size #[radians]

        assert nchan > 0, "must have a positive number of channels"
        self.nchan = int(nchan)

        assert (
            len(velocity_axis) == self.nchan
        ), "Velocity axis must be a list of length `nchan`"
        self.velocity_axis = torch.tensor(velocity_axis)

        img_radius = self.cell_size * (self.npix // 2)  # [radians]
        # calculate the image axes
        self.ll = torch.tensor(gridding.fftspace(img_radius, self.npix))  # [radians]
        # mm is the same

        # the output spatial frequencies of the RFFT routine (unshifted)
        self.us = torch.tensor(
            np.fft.rfftfreq(self.npix, d=self.cell_size) * 1e-3
        )  # convert to [kλ]
        self.vs = torch.tensor(
            np.fft.fftfreq(self.npix, d=self.cell_size) * 1e-3
        )  # convert to [kλ]

        # This shouldn't really be accessed by the user, since it's naturally
        # packed in the fftshifted format to make the Fourier transformation easier
        # and with East pointing right (i.e., RA increasing to the right)
        # this is contrary to the way astronomers normally plot images, but
        # is correct for what the FFT expects
        if cube is None:
            self._cube = nn.Parameter(
                torch.ones(
                    self.nchan,
                    self.npix,
                    self.npix,
                    requires_grad=True,
                    dtype=torch.double,
                )
            )
        else:
            self._cube = nn.Parameter(cube)

        # the image units are Jy/arcsec^2. An extended source with a brightness temperature
        # of 100 K is about 4 Jy/arcsec^2. These choice of units helps prevent
        # loss of numerical precision (I think)

        # calculate the pre-fftshifted gridding correction function
        self.corrfun = torch.tensor(
            gridding.corrfun_mat(np.fft.ifftshift(self.ll), np.fft.ifftshift(self.ll))
        )

        self.precached = False

    def precache_interpolation(self, dataset):
        """
        Caches the interpolation matrices used to interpolate the output from the FFT to the measured (u,v) points if the dataset has not been pre-gridded.
        Will be run automatically upon the first call to the model.

        Args:
            dataset: a UVDataset containing the u,v sampling points of the observation.

        Returns:
            None. Stores attributes self.C_res and self.C_ims, which are lists of sparse interpolation matrices corresponding to each channel.
        """

        max_baseline = torch.max(
            torch.abs(torch.cat([dataset.uu, dataset.vv]))
        )  # klambda

        # check that the pixel scale is sufficiently small to sample
        # the frequency corresponding to the largest baseline of the
        # dataset (in klambda)
        assert max_baseline < (
            1e-3 / (2 * self.cell_size)
        ), "Image cell size is too coarse to represent the largest spatial frequency sampled by the dataset. Make a finer image by decreasing cell_size. You may also need to increase npix to make sure the image remains wide enough to capture all of the emission and avoid aliasing."

        # calculate the interpolation matrices at the datapoints
        # the .detach().cpu() is to enable the numpy conversion even after transferred to GPU
        uu = dataset.uu.detach().cpu().numpy()
        vv = dataset.vv.detach().cpu().numpy()
        us = self.us.detach().cpu().numpy()
        vs = self.vs.detach().cpu().numpy()
        self.C_res = []
        self.C_ims = []
        for i in range(self.nchan):
            C_re, C_im = gridding.calc_matrices(uu[i], vv[i], us, vs)
            C_shape = C_re.shape

            # make these torch sparse tensors
            i_re = torch.LongTensor([C_re.row, C_re.col])
            v_re = torch.DoubleTensor(C_re.data)
            C_re = torch.sparse.DoubleTensor(i_re, v_re, torch.Size(C_shape))
            self.C_res.append(C_re)

            i_im = torch.LongTensor([C_im.row, C_im.col])
            v_im = torch.DoubleTensor(C_im.data)
            C_im = torch.sparse.DoubleTensor(i_im, v_im, torch.Size(C_shape))
            self.C_ims.append(C_im)

        self.precached = True

    def forward(self, dataset):
        """
        Compute the interpolated visibilities.

        Args:
            dataset (UVDataset): the dataset to forward model.
        """

        if dataset.gridded:
            # re, im output will always be 1D
            assert torch.allclose(
                dataset.uu, self.us
            ), "Pre-gridded uu is different than model us"
            assert torch.allclose(
                dataset.vv, self.vs
            ), "Pre-gridded vv is different than model vs"
            assert (
                dataset.npix == self.npix
            ), "Pre-gridded npix is different than model npix"
            assert (
                dataset.cell_size == self.cell_size
            ), "Pre-gridded cell_size is different than model cell_size."

            # convert the image to Jy/ster
            # and perform the RFFT
            self.vis = self.cell_size ** 2 * torch.rfft(
                self._cube / arcsec ** 2, signal_ndim=2
            )

            # torch delivers the real and imag components separately
            vis_re = self.vis[:, :, :, 0]
            vis_im = self.vis[:, :, :, 1]

            # grid mask is a (nchan, npix, npix//2 + 1) size boolean array
            re = vis_re.masked_select(dataset.grid_mask)
            im = vis_im.masked_select(dataset.grid_mask)

        else:
            # re, im output will always be 2D (nchan, nvis)

            # test to see if the interpolation is pre-cached
            if not self.precached:
                self.precache_interpolation(dataset)

            # TODO: does the corrfun broadcast correctly across the cube?
            self.vis = self.cell_size ** 2 * torch.rfft(
                self._cube * self.corrfun / arcsec ** 2, signal_ndim=2
            )

            # torch delivers the real and imag components separately
            vis_re = self.vis[:, :, :, 0]
            vis_im = self.vis[:, :, :, 1]

            # reshape into (nchan, -1, 1) vector format so we can do matrix product
            vr = torch.reshape(vis_re, (self.nchan, -1, 1))
            vi = torch.reshape(vis_im, (self.nchan, -1, 1))

            res = []
            ims = []
            # sample the FFT using the sparse matrices
            # the output of mm is a (nvis, 1) dimension tensor
            for i in range(self.nchan):
                res.append(torch.sparse.mm(self.C_res[i], vr[i]))
                ims.append(torch.sparse.mm(self.C_ims[i], vi[i]))

            # concatenate to a single (nchan, nvis) tensor
            re = torch.transpose(torch.cat(res, dim=1), 0, 1)
            im = torch.transpose(torch.cat(ims, dim=1), 0, 1)

        return re, im

    @property
    def cube(self):
        """
        Query the current state of the image.

        Returns:
            (2d numpy array)
        """
        # fftshift the image cube to the correct quadrants
        return mpol.utils.fftshift(self._cube, axes=(1, 2))

    @property
    def cube_detached(self):
        return self.cube.detach()

    @property
    def vis_cube(self):
        """
        Return the visibility cube and u, v axes.

        Returns:
            3-tuple of (us, vs, vis)
        """

        return (self.us, self.vs, self.vis)

    @property
    def extent(self):
        """
        Return the extent tuple (in arcsec) used for matplotlib plotting with imshow. Assumes `origin="upper"`.
        """
        low, high = (
            torch.min(self.ll) / arcsec,
            torch.max(self.ll) / arcsec,
        )  # [arcseconds]
        return [high, low, low, high]

    def to_FITS(self, fname="cube.fits", overwrite=False):
        """
        Export the image cube to a FITS file.

        Args:
            fname: the name of the FITS file to export to.
            overwrite: if the file already exists, overwrite?
        """

        try:
            from astropy.io import fits
        except ImportError:
            print(
                "Please install the astropy package to use FITS export functionality."
            )

        raise NotImplementedError

