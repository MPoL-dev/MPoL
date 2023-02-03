import numpy as np
import logging
import torch

from mpol.precomposed import SimpleNet
from mpol.training import TrainTest

class CrossValidate:
    r"""
    Utilities to run a cross-validation loop (implicitly running a training
    optimization loop), in order to compare MPoL models with different 
    hyperparameter values

    Parameters
    ----------
    coords : `mpol.coordinates.GridCoords` object
        Instance of the `mpol.coordinates.GridCoords` class.
    gridder : `mpol.gridding.Gridder` object
        Instance of the `mpol.gridding.Gridder` class.
    kfolds : int, default=5
        Number of k-folds to use in cross-validation
    split_method : str, default='random cell'
        Method to split full dataset into train/test subsets
    seed : int, default=None 
        Seed for random number generator used in splitting data
    learn_rate : float, default=0.5
        Neural network learning rate
    epochs : int, default=500
        Number of training iterations
    convergence_tol : float, default=1e-2
        Tolerance for training iteration stopping criterion as assessed by 
        loss function (suggested <= 1e-2)
    lambda_guess : list of str, default=None
        List of regularizers for which to guess an initial value 
    lambda_guess_briggs : list of float, default=[0.0, 0.5]
        Briggs robust values for two images used to guess initial regularizer 
        values (if lambda_guess is not None)
    lambda_entropy : float
        Relative strength for entropy regularizer
    entropy_prior_intensity : float, default=1e-10
        Prior value :math:`p` to calculate entropy against (suggested <<1)
    lambda_sparsity : float, default=None 
        Relative strength for sparsity regularizer
    lambda_TV : float, default=None
        Relative strength for total variation (TV) regularizer
    TV_epsilon : float, default=1e-10
        Softening parameter for TV regularizer (suggested <<1)
    lambda_TSV : float, default=None
        Relative strength for total squared variation (TSV) regularizer
    train_diag_step : int, default=None
        Interval at which optional training diagnostics are output
    diag_fig_train : bool, default=False
        Whether to generate a diagnostic figure during training
        (if True, `train_diag_step` must also be nonzero)
    device : torch.device, default=None
        Which hardware device to perform operations on (e.g., 'cuda:0').
        'None' defaults to current device. 
    verbose : bool, default=True
        Whether to print notification messages. 
    """
    def __init__(self, coords, gridder, kfolds=5, split_method='random_cell', 
                seed=None, learn_rate=0.5, 
                epochs=500, convergence_tol=1e-2, 
                lambda_guess=None, lambda_guess_briggs=[0.0, 0.5], 
                lambda_entropy=None, entropy_prior_intensity=1e-10, 
                lambda_sparsity=None, lambda_TV=None, 
                TV_epsilon=1e-10, lambda_TSV=None, 
                train_diag_step=None, diag_fig_train=False, device=None, 
                verbose=True):
        self._coords = coords
        self._gridder = gridder        
        self._kfolds = kfolds
        self._split_method = split_method
        self._seed = seed
        self._learn_rate = learn_rate
        self._epochs = epochs
        self._convergence_tol = convergence_tol
        self._lambda_guess = lambda_guess
        self._lambda_guess_briggs = lambda_guess_briggs
        self._lambda_entropy = lambda_entropy
        self._entropy_prior_intensity = entropy_prior_intensity
        self._lambda_sparsity = lambda_sparsity
        self._lambda_TV = lambda_TV
        self._TV_epsilon = TV_epsilon
        self._lambda_TSV = lambda_TSV
        self._train_diag_step = train_diag_step
        self._diag_fig_train = diag_fig_train
        self._device = device
        self._verbose = verbose


    def split_dataset(self, dataset):
        r"""
        Split a dataset into training and test subsets. 

        Parameters
        ----------
        dataset : PyTorch dataset object
            Instance of the `mpol.datasets.GriddedDataset` class

        Returns
        -------
        test_train_datasets : list of `mpol.datasets.GriddedDataset` objects
            Training and test subsets obtained from splitting the input dataset
        """
        # create a radial and azimuthal partition for the dataset
        dartboard = Dartboard(coords=self._coords)

        # use 'dartboard' to split full dataset into train/test subsets
        subsets = KFoldCrossValidatorGridded(dataset, k=self._kfolds,
                                        dartboard=dartboard,
                                        npseed=self._seed)

        if hasattr(self._device,'type') and self._device.type == 'cuda': # TODO: confirm which objects need to be passed to gpu
            test_train_datasets = [(train.to(self._device), test.to(self._device)) for (train, test) in subsets]
        else:
            test_train_datasets = [(train_pair, test_pair) for (train_pair, test_pair) in subsets]

        return test_train_datasets


    def run_crossval(self, test_train_datasets):
        r"""
        Run a cross-validation loop for a model obtained with a given set of 
        hyperparameters.

        Parameters
        ----------
        test_train_datasets : list of `mpol.datasets.GriddedDataset` objects
            Training and test subsets pre-split from the true dataset

        Returns
        -------
        cv_score : float 
            Mean cross-validation score across all k-folds
        all_scores : list of float
            Individual cross-validation scores for each k-fold 
        loss_histories : list of float 
            Loss function values for each training loop
        """
        loss_histories = []
        all_scores = []

        for kfold, (train_subset, test_subset) in enumerate(test_train_datasets):
            if self._verbose:
                logging.info("\nCross-validation: K-fold {} of {}".format(kfold, np.shape(test_train_datasets)[0] - 1))

            # create a new model and optimizer for this k_fold
            model = SimpleNet(coords=self._coords, nchan=train_subset.nchan)
            if hasattr(self._device,'type') and self._device.type == 'cuda': # TODO: confirm which objects need to be passed to gpu
                model = model.to(self._device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self._learn_rate)

            trainer = TrainTest(gridder=self._gridder, 
                                optimizer=optimizer, 
                                epochs=self._epochs, 
                                convergence_tol=self._convergence_tol, 
                                lambda_guess=self._lambda_guess,
                                lambda_guess_briggs=self._lambda_guess_briggs, 
                                lambda_entropy=self._lambda_entropy,
                                entropy_prior_intensity=self._entropy_prior_intensity,
                                lambda_sparsity=self._lambda_sparsity,
                                lambda_TV=self._lambda_TV, 
                                TV_epsilon=self._TV_epsilon,
                                lambda_TSV=self._lambda_TSV,
                                train_diag_step=self._train_diag_step, 
                                diag_fig_train=self._diag_fig_train,
                                verbose=self._verbose
            )

            loss, loss_history = trainer.train(model, train_subset)
            loss_histories.append(loss_history)
            all_scores.append(trainer.test(model, test_subset))

        # average individual test scores as a cross-val metric for chosen 
        # hyperparameters
        cv_score = np.mean(all_scores)

        return cv_score, all_scores, loss_histories
    

class Dartboard:
    r"""
    A polar coordinate grid relative to a :class:`~mpol.coordinates.GridCoords` object, reminiscent of a dartboard layout. The main utility of this object is to support splitting a dataset along radial and azimuthal bins for k-fold cross validation.

    Args:
        cell_size (float): the width of a pixel [arcseconds]
        npix (int): the number of pixels per image side
        coords (GridCoords): an object already instantiated from the GridCoords class. If providing this, cannot provide ``cell_size`` or ``npix``.
        q_edges (1D numpy array): an array of radial bin edges to set the dartboard cells in :math:`[\mathrm{k}\lambda]`. If ``None``, defaults to 12 log-linearly radial bins stretching from 0 to the :math:`q_\mathrm{max}` represented by ``coords``.
        phi_edges (1D numpy array): an array of azimuthal bin edges to set the dartboard cells in [radians], over the domain :math:`[0, \pi]`, which is also implicitly mapped to the domain :math:`[-\pi, \pi]` to preserve the Hermitian nature of the visibilities. If ``None``, defaults to 8 equal-spaced azimuthal bins stretched from :math:`0` to :math:`\pi`.

    """

    def __init__(
        self, cell_size=None, npix=None, coords=None, q_edges=None, phi_edges=None
    ):

        _setup_coords(self, cell_size, npix, coords)

        # copy over relevant quantities from coords
        # these are in packed format
        self.cartesian_qs = self.coords.packed_q_centers_2D
        self.cartesian_phis = self.coords.packed_phi_centers_2D

        # set q_max to the max q in coords
        self.q_max = self.coords.q_max  # [klambda]

        if q_edges is not None:
            self.q_edges = q_edges
        else:
            # set q edges approximately following inspriation from Petry et al. scheme:
            # https://ui.adsabs.harvard.edu/abs/2020SPIE11449E..1DP/abstract
            # first two bins set to 7m width
            # after third bin, bin width increases linearly until it is 700m at 16km baseline.
            # From 16m to 16km, bin width goes from 7m to 700m.

            # We aren't doing quite the same thing, just logspacing with a few linear cells at the start.
            self.q_edges = loglinspace(0, self.q_max, N_log=8, M_linear=5)

        if phi_edges is not None:
            assert np.all(phi_edges >= 0) & np.all(
                phi_edges <= np.pi
            ), "phi edges must be between 0 and pi"
            self.phi_edges = phi_edges
        else:
            # set phi edges
            self.phi_edges = np.linspace(0, np.pi, num=8 + 1)  # [radians]

    def get_polar_histogram(self, qs, phis):
        r"""
        Calculate a histogram in polar coordinates, using the bin edges defined by ``q_edges`` and ``phi_edges`` during initialization.

        Data coordinates should include the points for the Hermitian visibilities.

        Args:
            qs: 1d array of q values :math:`[\mathrm{k}\lambda]`
            phis: 1d array of datapoint azimuth values [radians] (must be the same length as qs)

        Returns:
            2d integer numpy array of cell counts, i.e., how many datapoints fell into each dartboard cell.

        """

        # make a polar histogram
        H, x_edges, y_edges = np.histogram2d(
            qs, phis, bins=[self.q_edges, self.phi_edges]
        )

        return H

    def get_nonzero_cell_indices(self, qs, phis):
        r"""
        Return a list of the cell indices that contain data points, using the bin edges defined by ``q_edges`` and ``phi_edges`` during initialization.

        Data coordinates should include the points for the Hermitian visibilities.

        Args:
            qs: 1d array of q values :math:`[\mathrm{k}\lambda]`
            phis: 1d array of datapoint azimuth values [radians] (must be the same length as qs)

        Returns:
            list of cell indices where cell contains at least one datapoint.
        """

        # make a polar histogram
        H = self.get_polar_histogram(qs, phis)

        indices = np.argwhere(H > 0)  # [i,j] indexes to go to q, phi

        return indices

    def build_grid_mask_from_cells(self, cell_index_list):
        r"""
        Create a boolean mask of size ``(npix, npix)`` (in packed format) corresponding to the ``vis_gridded`` and ``weight_gridded`` quantities of the :class:`~mpol.datasets.GriddedDataset` .

        Args:
            cell_index_list (list): list or iterable containing [q_cell, phi_cell] index pairs to include in the mask.

        Returns: (numpy array) 2D boolean mask in packed format.
        """
        mask = np.zeros_like(self.cartesian_qs, dtype="bool")

        # uses about a Gb..., and this only 256x256
        for cell_index in cell_index_list:

            qi, pi = cell_index
            q_min, q_max = self.q_edges[qi : qi + 2]
            p0_min, p0_max = self.phi_edges[pi : pi + 2]
            # also include Hermitian values
            p1_min, p1_max = self.phi_edges[pi : pi + 2] - np.pi

            # whether or not the q and phi values of the coordinate array
            # fit in the q cell and *either of* the regular or Hermitian phi cell
            ind = (
                (self.cartesian_qs >= q_min)
                & (self.cartesian_qs < q_max)
                & (
                    ((self.cartesian_phis > p0_min) & (self.cartesian_phis <= p0_max))
                    | ((self.cartesian_phis > p1_min) & (self.cartesian_phis <= p1_max))
                )
            )

            mask[ind] = True

        return mask


class KFoldCrossValidatorGridded:
    r"""
    Split a GriddedDataset into :math:`k` non-overlapping chunks, internally partitioned by a Dartboard. Inherit the properties of the GriddedDataset. This object creates an iterator providing a (train, test) pair of :class:`~mpol.datasets.GriddedDataset` for each k-fold.

    Args:
        griddedDataset (:class:`~mpol.datasets.GriddedDataset`): instance of the gridded dataset
        k (int): the number of subpartitions of the dataset
        dartboard (:class:`~mpol.datasets.Dartboard`): a pre-initialized Dartboard instance. If ``dartboard`` is provided, do not provide ``q_edges`` or ``phi_edges``.
        q_edges (1D numpy array): an array of radial bin edges to set the dartboard cells in :math:`[\mathrm{k}\lambda]`. If ``None``, defaults to 12 log-linearly radial bins stretching from 0 to the :math:`q_\mathrm{max}` represented by ``coords``.
        phi_edges (1D numpy array): an array of azimuthal bin edges to set the dartboard cells in [radians]. If ``None``, defaults to 8 equal-spaced azimuthal bins stretched from :math:`0` to :math:`\pi`.
        npseed (int): (optional) numpy random seed to use for the permutation, for reproducibility

    Once initialized, iterate through the datasets like

    >>> cv = datasets.KFoldCrossValidatorGridded(dataset, k)
    >>> for (train, test) in cv: # iterate among k datasets
    >>> ... # working with the n-th slice of k datasets
    >>> ... # do operations with train dataset
    >>> ... # do operations with test dataset

    """

    def __init__(
        self,
        griddedDataset,
        k,
        dartboard=None,
        q_edges=None,
        phi_edges=None,
        npseed=None,
    ):

        self.griddedDataset = griddedDataset

        assert k > 0, "k must be a positive integer"
        self.k = k

        if dartboard is not None:
            assert (q_edges is None) and (
                phi_edges is None
            ), "If providing a Dartboard instance, do not provide q_edges and phi_edges parameters."
            self.dartboard = dartboard
        else:
            self.dartboard = Dartboard(
                coords=self.griddedDataset.coords, q_edges=q_edges, phi_edges=phi_edges
            )

        # 2D mask for any UV cells that contain visibilities
        # in *any* channel
        stacked_mask = np.any(self.griddedDataset.mask.detach().numpy(), axis=0)

        # get qs, phis from dataset and turn into 1D lists
        qs = self.griddedDataset.coords.packed_q_centers_2D[stacked_mask]
        phis = self.griddedDataset.coords.packed_phi_centers_2D[stacked_mask]

        # create the full cell_list
        self.cell_list = self.dartboard.get_nonzero_cell_indices(qs, phis)

        # partition the cell_list into k pieces
        # first, randomly permute the sequence to make sure
        # we don't get structured radial/azimuthal patterns
        if npseed is not None:
            np.random.seed(npseed)
        self.k_split_cell_list = np.array_split(
            np.random.permutation(self.cell_list), k
        )

    def __iter__(self):
        self.n = 0  # the current k-slice we're on
        return self

    def __next__(self):
        if self.n < self.k:
            k_list = self.k_split_cell_list.copy()
            cell_list_test = k_list.pop(self.n)

            # put the remaining indices back into a full list
            cell_list_train = np.concatenate(k_list)

            # create the masks for each cell_list
            train_mask = self.dartboard.build_grid_mask_from_cells(cell_list_train)
            test_mask = self.dartboard.build_grid_mask_from_cells(cell_list_test)

            # copy original dateset
            train = copy.deepcopy(self.griddedDataset)
            test = copy.deepcopy(self.griddedDataset)

            # and use these masks to limit new datasets to only unmasked cells
            train.add_mask(train_mask)
            test.add_mask(test_mask)

            self.n += 1
            return train, test
        else:
            raise StopIteration
