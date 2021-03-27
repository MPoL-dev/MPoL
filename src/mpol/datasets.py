import numpy as np
import torch
from torch.utils.data import Dataset
from . import spheroidal_gridding
from .constants import *
from .coordinates import GridCoords, _setup_coords
import .utils import loglinspace


class GriddedDataset:
    r"""
    Args:
        cell_size (float): the width of a pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        nchan (int): the number of channels in the image (default = 1).
        vis_gridded (torch complex): the gridded visibility data stored in a "packed" format (pre-shifted for fft)
        weight_gridded (torch double): the weights corresponding to the gridded visibility data, also in a packed format
        mask (torch boolean): a boolean mask to index the non-zero locations of ``vis_gridded`` and ``weight_gridded`` in their packed format.
        device (torch.device) : the desired device of the dataset. If ``None``, defalts to current device.


    After initialization, the GriddedDataset provides the non-zero cells of the gridded visibilities and weights as a 1D vector via the following instance variables. This means that any individual channel information has been collapsed.

    :ivar vis_indexed: 1D complex tensor of visibility data
    :ivar weight_indexd: 1D tensor of weight values

    If you index the output of the Fourier layer in the same manner using ``self.mask`` (as done internally within :class:`~mpol.connectors.DataConnector`), then the model and data visibilities can be directly compared using a loss function.
    """

    def __init__(
        self,
        cell_size=None,
        npix=None,
        coords=None,
        nchan=None,
        vis_gridded=None,
        weight_gridded=None,
        mask=None,
        device=None,
    ):

        _setup_coords(self, cell_size, npix, coords, nchan)

        self.vis_gridded = torch.tensor(vis_gridded, device=device)
        self.weight_gridded = torch.tensor(weight_gridded, device=device)
        self.mask = torch.tensor(mask, device=device)

        # pre-index the values
        # note that these are *collapsed* across all channels
        # 1D array
        self.vis_indexed = self.vis_gridded[self.mask]
        self.weight_indexed = self.weight_gridded[self.mask]


# custom dataset loader
class UVDataset(Dataset):

    """
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
        uu=None,
        vv=None,
        weights=None,
        data_re=None,
        data_im=None,
        cell_size=None,
        npix=None,
        device=None,
        **kwargs
    ):

        # assert that all vectors are the same shape
        shape = uu.shape
        for a in [vv, weights, data_re, data_im]:
            assert a.shape == shape, "All dataset inputs must be the same shape."

        if len(shape) == 1:
            uu = np.atleast_2d(uu)
            vv = np.atleast_2d(vv)
            data_re = np.atleast_2d(data_re)
            data_im = np.atleast_2d(data_im)
            weights = np.atleast_2d(weights)

        self.nchan = shape[0]

        assert np.all(
            weights > 0.0
        ), "Not all thermal weights are positive, check inputs."

        if cell_size is not None and npix is not None:
            self.cell_size = cell_size * arcsec  # [radians]
            self.npix = npix

            (
                uu_grid,
                vv_grid,
                grid_mask,
                g_weights,
                g_re,
                g_im,
            ) = spheroidal_gridding.grid_dataset(
                uu,
                vv,
                weights,
                data_re,
                data_im,
                self.cell_size / arcsec,
                npix=self.npix,
            )

            # grid_mask (nchan, npix, npix//2 + 1) bool: a boolean array the same size as the output of the RFFT, designed to directly index into the output to evaluate against pre-gridded visibilities.
            self.uu = torch.tensor(uu_grid, device=device)
            self.vv = torch.tensor(vv_grid, device=device)
            self.grid_mask = torch.tensor(grid_mask, dtype=torch.bool, device=device)
            self.weights = torch.tensor(g_weights, device=device)
            self.re = torch.tensor(g_re, device=device)
            self.im = torch.tensor(g_im, device=device)
            self.gridded = True

        else:
            self.gridded = False
            self.uu = torch.tensor(uu, dtype=torch.double, device=device)  # klambda
            self.vv = torch.tensor(vv, dtype=torch.double, device=device)  # klambda
            self.weights = torch.tensor(
                weights, dtype=torch.double, device=device
            )  # 1/Jy^2
            self.re = torch.tensor(data_re, dtype=torch.double, device=device)  # Jy
            self.im = torch.tensor(data_im, dtype=torch.double, device=device)  # Jy

        # TODO: store kwargs to do something for antenna self-cal

    def __getitem__(self, index):
        return (
            self.uu[index],
            self.vv[index],
            self.weights[index],
            self.re[index],
            self.im[index],
        )

    def __len__(self):
        return len(self.uu)


class GriddedDartboard:
    r"""
    Args:
        nq (int): number of radial bins
        qmax (float): maximum baseline (in klambda)
        nphi (int): number of azimuthal bins
    """

    def __init__(
        self, cell_size=None, npix=None, coords=None, nchan=None, nq=None, nphi=32
    ):

        _setup_coords(self, cell_size, npix, coords, nchan)

        # set q_max to the max q in coords
        self.q_max = self.coords.q_max  # [klambda]

        # set q edges approximately following Petry et al. scheme: https://ui.adsabs.harvard.edu/abs/2020SPIE11449E..1DP/abstract
        # first two bins set to 7m width
        # after third bin, bin width increases linearly until it is 700m at 16km baseline.
        # From 16m to 16km, bin width goes from 7m to 700m.
        
        # We aren't doing quite the same thing, just logspacing with a few linear cells at the start.
        self.q_edges = loglinspace(0, self.q_max, N_log=nq-2, M_linear=2)

        # set phi edges
        self.phi_edges = np.linspace(0, 2 * np.pi, num=nphi + 1)
        self.phi_centers = np.diff(self.phi_edges) + self.phi_edges[:-1]

        # create an index of cells q, phi.

    def data_cells(self, datamask):
        r"""
        Calculate the cell indices that contain data
        """
        pass
# Dartboard (init w/ nr, nphi, rmax):
#             store cell centers, cell edges
#             store coords obj
#         Dartboard data_cells(uv_pixel_mask or loose visibilities):
#             calculate cell index pairs that have data
#         Dartboard create_uv_mask(r_phi_indices):
#             given a list of r, phi indices, create a uv_pixel_mask
#             for cells that *have data* *and* are in these indices


# class KFoldCrossValidatorGridded:
#     r"""
#     Split a GriddeDataset into k non-overlapping chunks.

#     Split radially.

#     Args:
#         k (int): the number of subpartitions


#     """

#     def __init__(self, griddedDataset, k, npseed=None):

#         assert k > 0, "k must be a positive integer"
#         self.k = k

#         if npseed is not None:
#             np.random.seed(npseed)

#         # setup the dartboard to create r, phi polar cells (meshgrid?)
#         #
#         # compare dataset mask to dartboard and determine which polar cells have data
#         # store these as cell indices
#         #
#         # split this list of cells into k groups


#         # we really want pixel masks
#             # of which (r, phi) cells have (any) data
#             # of which cells are


#         # store the reference to the original dataset
#         self.griddedDataset = griddedDataset

#     def create_masks_from_cells(k_cell_list):
#         """

#         """
#         # modify the mask
#         # to create one with only those k cells
#         # and another with (k-1) cells
#         self.mask = torch.tensor(mask, device=device)

#         return train_mask, test_mask

#     def _get_nth_datasets(self, n):
#         """
#         Return the train and test datasets corresponding to the n-th slice through the k-folds.
#         """

#         # index the k cell list

#         #
#         # create a dataset with mask indexes only if they fall within the k group
#         # create a dataset containing all k-1 mask indices *except* those in the k group
#         pass

#     def __iter__(self):
#         self.n = 0  # the current k-slice we're on
#         return self

#     def __next__(self):
#         if self.n < k:
#             # TODO: index the k_cell_list
#             # TODO: calculate the train and test datasets
#             return train, test
#             self.n += 1
#         else:
#             raise StopIteration

