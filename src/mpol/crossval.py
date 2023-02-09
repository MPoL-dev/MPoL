from __future__ import annotations

import numpy as np
import copy
from collections import defaultdict
import logging
import torch

from mpol.precomposed import SimpleNet
from mpol.training import TrainTest
from mpol.datasets import Dartboard, GriddedDataset

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
    split_method : str, default='random_cell'
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
        if self._split_method == 'dartboard':
            # create a radial and azimuthal partition for the dataset
            dartboard = Dartboard(coords=self._coords)

            # use 'dartboard' to split full dataset into train/test subsets
            subsets = KFoldCrossValidatorGridded(dataset, k=self._kfolds,
                                            dartboard=dartboard,
                                            npseed=self._seed)

        elif self._split_method == 'random_cell':
            # get indices for the 20 cells with the highest binned weight
            top20 = np.argpartition(dataset.weight_indexed, -20)[-20:]
            vis_all = dataset.vis_indexed
            # subsets = # TODO

        else:
            supported_methods = ['dartboard, random_cell']
            raise ValueError("'split_method' {} must be one of "
                            "{}".format(split_method, supported_methods))

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


class RandomCellSplitGridded:
    r"""
    Split a GriddedDataset into :math:`k` subsets. Inherit the properties of 
    the GriddedDataset. This object creates an iterator providing a 
    (train, test) pair of :class:`~mpol.datasets.GriddedDataset` for each 
    k-fold.

    Parameters
    ----------
    dataset : PyTorch dataset object
        Instance of the `mpol.datasets.GriddedDataset` class
    kfolds : int, default=5
        Number of k-folds (partitions) of `dataset`
    seed : int, default=None 
        Seed for PyTorch random number generator used to shuffle data before
        splitting
    channel : int, default=0 
        Channel of the dataset to use in determining the splits

    Once initialized, iterate through the datasets like:
        >>> split_iterator = crossval.RandomCellSplitGridded(dataset, kfolds)
        >>> for (train, test) in split_iterator: # iterate through `kfolds` datasets
        >>> ... # working with the n-th slice of `kfolds` datasets
        >>> ... # do operations with train dataset
        >>> ... # do operations with test dataset    

    Notes
    -----
    Treats `dataset` as a single-channel object with all data in `channel`
    """

    def __init__(self, dataset, kfolds=5, seed=None, channel=0):
        self.dataset = dataset
        self.kfolds = kfolds
        self.channel = channel

        # get indices for cells in the top 1% of gridded weight 
        # (we'll want all training sets to have these high SNR points)
        nvis = len(self.dataset.vis_indexed)
        nn = int(nvis * 0.01)
        # get the nn-th largest value in weight_indexed
        w_thresh = np.partition(self.dataset.weight_indexed, -nn)[-nn]
        self._top_nn = torch.argwhere(self.dataset.weight_gridded[self.channel] >= w_thresh).T

        # mask these indices
        self.top_mask = torch.ones(self.dataset.weight_gridded[self.channel].shape, dtype=bool)
        self.top_mask[self._top_nn[0], self._top_nn[1]] = False
        # use unmasked cells that also have data for splits
        self.split_mask = torch.logical_and(self.dataset.mask[self.channel], self.top_mask)
        split_idx = torch.argwhere(self.split_mask).T 

        # shuffle indices to prevent radial/azimuthal patterns in splits
        if seed is not None:
            torch.manual_seed(seed) 
        shuffle = torch.randperm(split_idx.shape[1])
        split_idx = split_idx[:,shuffle] 

        # split indices into k subsets
        self.splits = torch.tensor_split(split_idx, self.kfolds, dim=1) 

    def __iter__(self):
        # current k-slice
        self._n = 0  
        return self

    def __next__(self): 
        if self._n < self.kfolds:
            test_idx = self.splits[self._n]
            train_idx = torch.cat(([self.splits[x] for x in range(len(self.splits)) if x != self._n]), dim=1)
            # add the masked (high SNR) points to the current training set 
            train_idx = torch.cat((train_idx, self._top_nn), dim=1) 

            train_mask = torch.zeros(self.dataset.weight_gridded[self.channel].shape, dtype=bool)
            test_mask = torch.zeros(self.dataset.weight_gridded[self.channel].shape, dtype=bool)            
            train_mask[train_idx[0], train_idx[1]] = True
            test_mask[test_idx[0], test_idx[1]] = True

            # copy original dataset
            train = copy.deepcopy(self.dataset)
            test = copy.deepcopy(self.dataset)

            # use the masks to limit new datasets to only unmasked cells
            train.add_mask(train_mask) 
            test.add_mask(test_mask)

            self._n += 1

            return train, test

        else:
            raise StopIteration
    

class DartboardSplitGridded:
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

    >>> cv = crossval.DartboardSplitGridded(dataset, k)
    >>> for (train, test) in cv: # iterate among k datasets
    >>> ... # working with the n-th slice of k datasets
    >>> ... # do operations with train dataset
    >>> ... # do operations with test dataset

    """

    def __init__(
        self,
        gridded_dataset: GriddedDataset,
        k: int,
        dartboard: Dartboard | None = None,
        npseed: int | None = None,
    ):
        if k <= 0:
            raise ValueError("k must be a positive integer")

        if dartboard is None:
            dartboard = Dartboard(coords=gridded_dataset.coords)

        self.griddedDataset = gridded_dataset
        self.k = k
            self.dartboard = dartboard

        # 2D mask for any UV cells that contain visibilities
        # in *any* channel
        stacked_mask = torch.any(self.griddedDataset.mask, dim=0)

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

    @classmethod
    def from_dartboard_properties(
        cls,
        gridded_dataset: GriddedDataset,
        k: int,
        q_edges: NDArray[floating[Any]],
        phi_edges: NDArray[floating[Any]],
        npseed: int | None = None,
    ) -> DartboardSplitGridded:
        """
        Alternative method to initialize a DartboardSplitGridded object from Dartboard parameters.

         Args:
             griddedDataset (:class:`~mpol.datasets.GriddedDataset`): instance of the gridded dataset
             k (int): the number of subpartitions of the dataset
             q_edges (1D numpy array): an array of radial bin edges to set the dartboard cells in :math:`[\mathrm{k}\lambda]`. If ``None``, defaults to 12 log-linearly radial bins stretching from 0 to the :math:`q_\mathrm{max}` represented by ``coords``.
             phi_edges (1D numpy array): an array of azimuthal bin edges to set the dartboard cells in [radians]. If ``None``, defaults to 8 equal-spaced azimuthal bins stretched from :math:`0` to :math:`\pi`.
             npseed (int): (optional) numpy random seed to use for the permutation, for reproducibility
        """
        dartboard = Dartboard(gridded_dataset.coords, q_edges, phi_edges)
        return cls(gridded_dataset, k, dartboard, npseed)

    def __iter__(self) -> DartboardSplitGridded:
        self.n = 0  # the current k-slice we're on
        return self

    def __next__(self) -> tuple[GriddedDataset, GriddedDataset]:
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
