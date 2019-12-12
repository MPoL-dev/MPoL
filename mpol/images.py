import numpy as np
import torch
from torch import nn

from mpol import gridding
from mpol.constants import *


class MpolImage(nn.Module):
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
        # fliplr so that East is to the left
        return np.fft.fftshift(self._image.detach().numpy())

    @property
    def extent(self):
        """
        Return the extent tuple (in arcsec) used for matplotlib plotting with imshow. Assumes `origin="upper"`.
        """
        low, high = np.min(self.ll) / arcsec, np.max(self.ll) / arcsec  # [arcseconds]
        return [high, low, low, high]
