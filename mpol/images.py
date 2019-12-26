import numpy as np
import torch
from torch import nn

from mpol import gridding
from mpol.constants import *
import mpol.utils


class Image(nn.Module):
    def __init__(
        self,
        npix=None,
        cell_size=None,
        image=None,
        dataset=None,
        grid_indices=None,
        **kwargs
    ):
        """
        Initialize a Model class.

        Args:
            npix: the number of pixels per image side
            cell_size: the size of a pixel in arcseconds
            image: an image to initialize the model with. If None, assumes image is all ones.
            dataset (UVDataset): the dataset to precache the interpolation matrices against.
        """
        super().__init__()
        assert npix % 2 == 0, "npix must be even (for now)"
        self.npix = npix

        assert cell_size > 0.0, "cell_size must be positive (arcseconds)"
        self.cell_size = cell_size * arcsec  # [radians]
        # cell_size is also the differential change in sky angles
        # dll = dmm = cell_size #[radians]

        img_radius = self.cell_size * (self.npix // 2)  # [radians]
        # calculate the image axes
        self.ll = gridding.fftspace(img_radius, self.npix)  # [radians]
        # mm is the same

        # the output spatial frequencies of the RFFT routine (unshifted)
        self.us = np.fft.rfftfreq(self.npix, d=self.cell_size) * 1e-3  # convert to [kλ]
        self.vs = np.fft.fftfreq(self.npix, d=self.cell_size) * 1e-3  # convert to [kλ]

        # This shouldn't really be accessed by the user, since it's naturally
        # packed in the fftshifted format to make the Fourier transformation easier
        # and with East pointing right (i.e., RA increasing to the right)
        # this is contrary to the way astronomers normally plot images, but
        # is correct for what the FFT expects
        if image is None:
            self._image = nn.Parameter(
                # torch.zeros(self.npix, self.npix, requires_grad=True, dtype=torch.double)
                torch.ones(self.npix, self.npix, requires_grad=True, dtype=torch.double)
            )
        else:
            self._image = nn.Parameter(
                # torch.zeros(self.npix, self.npix, requires_grad=True, dtype=torch.double)
                torch.tensor(image, requires_grad=True, dtype=torch.double)
            )
        # the units are Jy/arcsec^2. An extended source with a brightness temperature
        # of 100 K is about 4 Jy/arcsec^2. These choice of units helps prevent
        # loss of numerical precision (I think)

        # calculate the pre-fftshifted gridding correction function
        self.corrfun = torch.tensor(
            gridding.corrfun_mat(np.fft.ifftshift(self.ll), np.fft.ifftshift(self.ll))
        )

        self.gridded = False

        if dataset is not None:
            self.precache_interpolation(dataset)
        else:
            self.grid_indices = grid_indices
            self.gridded = True

    def precache_interpolation(self, dataset):
        """
        Caches the interpolation matrices used to interpolate the output from the FFT to the measured (u,v) points.
        If you did not specify your dataset when instantiating the model, run this before calculating a loss.

        Args:
            dataset: a UVDataset containing the u,v sampling points of the observation.

        Returns:
            None. Stores attributes self.C_re and self.C_im
        """

        max_baseline = np.max(
            np.abs([dataset.uu.numpy(), dataset.vv.numpy()])
        )  # klambda

        # check that the pixel scale is sufficiently small to sample
        # the frequency corresponding to the largest baseline of the
        # dataset (in klambda)
        assert max_baseline < (
            1e-3 / (2 * self.cell_size)
        ), "Image cell size is too coarse to represent the largest spatial frequency sampled by the dataset. Make a finer image by decreasing cell_size. You may also need to increase npix to make sure the image remains wide enough."

        # calculate the interpolation matrices at the datapoints
        C_re, C_im = gridding.calc_matrices(
            dataset.uu.numpy(), dataset.vv.numpy(), self.us, self.vs
        )
        C_shape = C_re.shape

        # make these torch sparse tensors
        i_re = torch.LongTensor([C_re.row, C_re.col])
        v_re = torch.DoubleTensor(C_re.data)
        self.C_re = torch.sparse.DoubleTensor(i_re, v_re, torch.Size(C_shape))

        i_im = torch.LongTensor([C_im.row, C_im.col])
        v_im = torch.DoubleTensor(C_im.data)
        self.C_im = torch.sparse.DoubleTensor(i_im, v_im, torch.Size(C_shape))

    def forward(self):
        """
        Compute the interpolated visibilities.
        """

        # get the RFFT'ed values
        # image is converted to Jy/ster

        if self.gridded:
            vis = self.cell_size ** 2 * torch.rfft(
                self._image / arcsec ** 2, signal_ndim=2
            )

            # torch delivers the real and imag components separately
            vis_re = vis[:, :, 0]
            vis_im = vis[:, :, 1]

            re = vis_re[self.grid_indices]
            im = vis_im[self.grid_indices]

        else:
            vis = self.cell_size ** 2 * torch.rfft(
                self._image * self.corrfun / arcsec ** 2, signal_ndim=2
            )

            # torch delivers the real and imag components separately
            vis_re = vis[:, :, 0]
            vis_im = vis[:, :, 1]

            # reshape into (-1, 1) vector format so we can do matrix product
            vr = torch.reshape(vis_re, (-1, 1))
            vi = torch.reshape(vis_im, (-1, 1))

            # sample the FFT using the sparse matrices
            # also trim the last dimension so that these are 1D tensors
            re = torch.sparse.mm(self.C_re, vr)[:, 0]
            im = torch.sparse.mm(self.C_im, vi)[:, 0]

        return re, im

    @property
    def image(self):
        """
        Query the current state of the image.

        Returns:
            (2d numpy array)
        """
        # get the image
        # fftshift it to the correct quadrants
        return mpol.utils.fftshift(self._image)

    @property
    def image_detached(self):
        return self.image.detach()

    @property
    def extent(self):
        """
        Return the extent tuple (in arcsec) used for matplotlib plotting with imshow. Assumes `origin="upper"`.
        """
        low, high = np.min(self.ll) / arcsec, np.max(self.ll) / arcsec  # [arcseconds]
        return [high, low, low, high]

    def to_FITS(self, fname="image.fits", overwrite=False):
        """
        Export the image to a FITS file.

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

        pass


class ImageCube(nn.Module):
    def __init__(
        self,
        npix=None,
        cell_size=None,
        nchan=None,
        velocity_axis=None,
        cube=None,
        dataset=None,
        grid_mask=None,
        **kwargs
    ):
        """
        Initialize an ImageCube.

        Args:
            npix (int): the number of pixels per image side
            cell_size (float): the size of a pixel in arcseconds
            nchan (int): the number of channels in the image
            velocity_axis (list): vector of velocities (in km/s) corresponding to nchan. Channels should be spaced approximately equidistant in velocity but need not be strictly exact.
            cube (PyTorch tensor w/ `requires_grad = True`): an image cube to initialize the model with. If None, assumes cube is all ones.
            dataset (UVDataset): the dataset to precache the interpolation matrices against.
            grid_mask (nchan, npix, npix//2 + 1) bool: a boolean array the same size as the output of the RFFT, designed to directly index into the output to evaluate against pre-gridded visibilities.
        """
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
        self.velocity_axis = velocity_axis

        img_radius = self.cell_size * (self.npix // 2)  # [radians]
        # calculate the image axes
        self.ll = gridding.fftspace(img_radius, self.npix)  # [radians]
        # mm is the same

        # the output spatial frequencies of the RFFT routine (unshifted)
        self.us = np.fft.rfftfreq(self.npix, d=self.cell_size) * 1e-3  # convert to [kλ]
        self.vs = np.fft.fftfreq(self.npix, d=self.cell_size) * 1e-3  # convert to [kλ]

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

        self.gridded = False

        if dataset is not None:
            self.precache_interpolation(dataset)
        else:
            self.grid_mask = torch.tensor(grid_mask, dtype=torch.bool)
            self.gridded = True

    def precache_interpolation(self, dataset):
        """
        Caches the interpolation matrices used to interpolate the output from the FFT to the measured (u,v) points.
        If you did not specify your dataset when instantiating the model, run this before calculating a loss.

        Args:
            dataset: a UVDataset containing the u,v sampling points of the observation.

        Returns:
            None. Stores attributes self.C_re and self.C_im
        """
        raise NotImplementedError

        max_baseline = np.max(
            np.abs([dataset.uu.numpy(), dataset.vv.numpy()])
        )  # klambda

        # check that the pixel scale is sufficiently small to sample
        # the frequency corresponding to the largest baseline of the
        # dataset (in klambda)
        assert max_baseline < (
            1e-3 / (2 * self.cell_size)
        ), "Image cell size is too coarse to represent the largest spatial frequency sampled by the dataset. Make a finer image by decreasing cell_size. You may also need to increase npix to make sure the image remains wide enough to capture all of the emission and avoid aliasing."

        # calculate the interpolation matrices at the datapoints
        C_re, C_im = gridding.calc_matrices(
            dataset.uu.numpy(), dataset.vv.numpy(), self.us, self.vs
        )
        C_shape = C_re.shape

        # make these torch sparse tensors
        i_re = torch.LongTensor([C_re.row, C_re.col])
        v_re = torch.DoubleTensor(C_re.data)
        self.C_re = torch.sparse.DoubleTensor(i_re, v_re, torch.Size(C_shape))

        i_im = torch.LongTensor([C_im.row, C_im.col])
        v_im = torch.DoubleTensor(C_im.data)
        self.C_im = torch.sparse.DoubleTensor(i_im, v_im, torch.Size(C_shape))

    def forward(self):
        """
        Compute the interpolated visibilities.
        """

        if self.gridded:

            # convert the image to Jy/ster
            # and perform the RFFT
            vis = self.cell_size ** 2 * torch.rfft(
                self._cube / arcsec ** 2, signal_ndim=2
            )

            # torch delivers the real and imag components separately
            vis_re = vis[:, :, :, 0]
            vis_im = vis[:, :, :, 1]

            # grid mask is a (nchan, npix, npix//2 + 1) size boolean array
            re = vis_re.masked_select(self.grid_mask)
            im = vis_im.masked_select(self.grid_mask)

        else:
            raise NotImplementedError
            # TODO: does the corrfun broadcast correctly?
            vis = self.cell_size ** 2 * torch.rfft(
                self._cube * self.corrfun / arcsec ** 2, signal_ndim=2
            )

            # torch delivers the real and imag components separately
            vis_re = vis[:, :, :, 0]
            vis_im = vis[:, :, :, 1]

            # TODO: figure out the fastest way to do the sparse matrix multiply
            # I think we want to preserve the capacity to have individual channel
            # visibilities at the end, so probably a per-channel multiply makes
            # the most sense

            # reshape into (-1, 1) vector format so we can do matrix product
            vr = torch.reshape(vis_re, (-1, 1))
            vi = torch.reshape(vis_im, (-1, 1))

            # sample the FFT using the sparse matrices
            # also trim the last dimension so that these are 1D tensors
            re = torch.sparse.mm(self.C_re, vr)[:, 0]
            im = torch.sparse.mm(self.C_im, vi)[:, 0]

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
    def extent(self):
        """
        Return the extent tuple (in arcsec) used for matplotlib plotting with imshow. Assumes `origin="upper"`.
        """
        low, high = np.min(self.ll) / arcsec, np.max(self.ll) / arcsec  # [arcseconds]
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

