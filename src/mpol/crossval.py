from __future__ import annotations

import copy
import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from numpy import floating
from numpy.typing import NDArray

from mpol.datasets import Dartboard, GriddedDataset
from mpol.precomposed import SimpleNet
from mpol.training import TrainTest, train_to_dirty_image
from mpol.plot import split_diagnostics_fig
from mpol.utils import loglinspace

class CrossValidate:
    r"""
    Utilities to run a cross-validation loop (implicitly running a training
    optimization loop), in order to compare MPoL models with different
    hyperparameter values

    Parameters
    ----------
    coords : `mpol.coordinates.GridCoords` object
        Instance of the `mpol.coordinates.GridCoords` class.
    imager : `mpol.gridding.DirtyImager` object
        Instance of the `mpol.gridding.DirtyImager` class.
    learn_rate : float, default=0.3
        Initial learning rate  
    regularizers : nested dict, default={}
        Dictionary of image regularizers to use. For each, a dict of the 
        strength ('lambda', float), whether to guess an initial value for lambda 
        ('guess', bool), and other quantities needed to compute their loss term.
        Example:
        {"sparsity":{"lambda":1e-3, "guess":False},
        "entropy": {"lambda":1e-3, "guess":True, "prior_intensity":1e-10}
        }
    epochs : int, default=10000
        Number of training iterations
    convergence_tol : float, default=1e-5
        Tolerance for training iteration stopping criterion as assessed by
        loss function (suggested <= 1e-3)
    schedule_factor : float, default=0.995
        For the `torch.optim.lr_scheduler.ReduceLROnPlateau` scheduler, factor 
        to which the learning rate is reduced when learning rate stops decreasing
    start_dirty_image : bool, default=False
        Whether to start the RML optimization loop by initializing the model 
        image to a dirty image of the observed data. If False, the optimization
        loop will start with a blank image.
    train_diag_step : int, default=None
        Interval at which training diagnostics are output. If None, no
        diagnostics will be generated.
    kfolds : int, default=5
        Number of k-folds to use in cross-validation
    split_method : str, default='dartboard'
        Method to split full dataset into train/test subsets
    dartboard_q_edges, dartboard_phi_edges : list of float, default=None, unit=[klambda]
        Radial and azimuthal bin edges of the cells used to split the dataset
        if `split_method`==`dartboard` (see `datasets.Dartboard`)
    split_diag_fig : bool, default=False
        Whether to generate a diagnostic figure of dataset splitting into
        train/test sets.
    store_cv_diagnostics : bool, default=False
        Whether to store diagnostics of the cross-validation loop.
    save_prefix : str, default=None
        Prefix (path) used for saved figure names. If None, figures won't be
        saved
    verbose : bool, default=True
        Whether to print notification messages.
    device : torch.device, default=None
        Which hardware device to perform operations on (e.g., 'cuda:0').
        'None' defaults to current device.        
    seed : int, default=None
        Seed for random number generator used in splitting data        
    """

    def __init__(self, coords, imager, learn_rate=0.3, 
                regularizers={}, epochs=10000, convergence_tol=1e-5, 
                schedule_factor=0.995,
                start_dirty_image=False, 
                train_diag_step=None, kfolds=5, split_method="dartboard",
                dartboard_q_edges=None, dartboard_phi_edges=None,
                split_diag_fig=False, store_cv_diagnostics=False, 
                save_prefix=None, verbose=True, device=None, seed=None,
                ):
        self._coords = coords
        self._imager = imager
        self._learn_rate = learn_rate
        self._regularizers = regularizers
        self._epochs = epochs
        self._convergence_tol = convergence_tol
        self._schedule_factor = schedule_factor
        self._start_dirty_image = start_dirty_image
        self._train_diag_step = train_diag_step
        self._kfolds = kfolds
        self._split_method = split_method
        self._dartboard_q_edges = dartboard_q_edges
        self._dartboard_phi_edges = dartboard_phi_edges
        self._split_diag_fig = split_diag_fig
        self._store_cv_diagnostics = store_cv_diagnostics
        self._save_prefix = save_prefix
        self._verbose = verbose
        self._device = device
        self._seed = seed

        self._split_figure = None

        # used to collect objects across all kfolds
        self._diagnostics = None

        if self._verbose:
            logging.info("\nCross-validation")

    def split_dataset(self, dataset):
        r"""
        Split a dataset into training and test subsets.

        Parameters
        ----------
        dataset : PyTorch dataset object
            Instance of the `mpol.datasets.GriddedDataset` class

        Returns
        -------
        split_iterator : iterator returning tuple
            Iterator that provides a (train, test) pair of 
            :class:`~mpol.datasets.GriddedDataset` for each k-fold
        """
        if self._split_method == "random_cell":
            split_iterator = RandomCellSplitGridded(
                dataset=dataset, k=self._kfolds, seed=self._seed
            )

        elif self._split_method == "dartboard":
            if self._dartboard_q_edges is None:
                # create a radial partition for the dataset.
                # this is the same as the default q_edges in `datasets.Dartboard`,
                # except that the max baseline is set by (a padding factor times)
                # the maximum baseline in the dataset, rather than by the largest
                # baseline in the Fourier plane grid `coords.q_max` (which can
                # often be a factor of >~2 larger than the longest baseline in
                # the dataset).
                stacked_mask = torch.any(dataset.mask, axis=0)
                stacked_mask = stacked_mask.to('cpu') # TODO: remove
                qs = dataset.coords.packed_q_centers_2D[stacked_mask]
                pad_factor = 1.1
                q_edges = loglinspace(0, qs.max() * pad_factor, N_log=8, M_linear=5)

            dartboard = Dartboard(coords=self._coords, 
                                  q_edges=q_edges, 
                                  phi_edges=self._dartboard_phi_edges,
                                  )

            # use 'dartboard' to split full dataset into train/test subsets
            split_iterator = DartboardSplitGridded(
                dataset, k=self._kfolds, dartboard=dartboard, seed=self._seed
            )

            if self._verbose:
                logging.info(f"  Max baseline in Fourier grid {self._coords.q_max:.0f} klambda")
                logging.info(f"    Dartboard: baseline bin edges {[round(x, 1) for x in dartboard.q_edges.tolist()]} klambda")

        else:
            supported_methods = ["dartboard", "random_cell"]
            raise ValueError(
                "'split_method' {} must be one of "
                "{}".format(self._split_method, supported_methods)
            )

        return split_iterator

    def run_crossval(self, dataset):
        r"""
        Run a cross-validation loop for a model obtained with a given set of
        hyperparameters.

        Parameters
        ----------
        dataset : dataset object
            Instance of the `mpol.datasets.GriddedDataset` class
        Returns
        -------
        cv_score : dict 
            Dictionary with mean and standard deviation of cross-validation 
            scores across all k-folds, and all raw scores
        """
        all_scores = []
        if self._store_cv_diagnostics:
            self._diagnostics = defaultdict(list)

        split_iterator = self.split_dataset(dataset)
        if self._split_diag_fig:
            split_fig, split_axes = split_diagnostics_fig(split_iterator, save_prefix=self._save_prefix)
            self._split_figure = (split_fig, split_axes)

        for kk, (train_set, test_set) in enumerate(split_iterator):
            if self._verbose:
                logging.info("\n  k-fold {} of {}".format(kk, self._kfolds - 1))

            # if hasattr(self._device,'type') and self._device.type == 'cuda': # TODO: confirm which objects need to be passed to gpu
            #     train_set, test_set = train_set.to(self._device), test_set.to(self._device)

            model = SimpleNet(coords=self._coords, nchan=self._imager.nchan)
            if self._start_dirty_image is True:
                if kk == 0:
                    if self._verbose:
                        logging.info(
                            "\n  Pre-training to dirty image to initialize subsequent optimization loops"
                        )
                    # initial short training loop to get model image to approximate dirty image
                    model_pretrained = train_to_dirty_image(model=model, imager=self._imager)
                    # save the model to a state we can load in subsequent kfolds
                    torch.save(model_pretrained.state_dict(), f=self._save_prefix + "_dirty_image_model.pt")
                else:
                    # create a new model for this kfold, initializing it to the model pretrained on the dirty image
                    model.load_state_dict(torch.load(self._save_prefix + "_dirty_image_model.pt"))

            # create a new optimizer and scheduler for this kfold
            optimizer = torch.optim.Adam(model.parameters(), lr=self._learn_rate)
            if self._schedule_factor is None:
                scheduler = None
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self._schedule_factor)

            trainer = TrainTest( 
                imager=self._imager,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=self._epochs,
                convergence_tol=self._convergence_tol,
                regularizers=self._regularizers,
                train_diag_step=self._train_diag_step,
                kfold=kk,
                save_prefix=self._save_prefix,
                verbose=self._verbose,
            )

            # run training 
            loss, loss_history = trainer.train(model, train_set)

            # run testing
            all_scores.append(trainer.test(model, test_set))

            # store objects from the most recent kfold for diagnostics           
            self._model = model
            self._train_figure = trainer.train_figure

            # collect objects from this kfold to store
            if self._store_cv_diagnostics:
                self._diagnostics["models"].append(self._model)
                self._diagnostics["regularizers"].append(self._regularizers)
                self._diagnostics["loss_histories"].append(loss_history)                
                # self._diagnostics["train_figures"].append(self._train_figure)

        # average individual test scores to get the cross-val metric for chosen
        # hyperparameters
        self._cv_score = {
            "mean": np.mean(all_scores),
            "std": np.std(all_scores),
            "all": all_scores,
        }
        
        return self._cv_score
    
    @property
    def model(self):
        """For the most recent kfold, trained model (`SimpleNet` class instance)"""
        return self._model

    @property
    def regularizers(self):
        """For the most recent kfold, dict containing regularizers used and their strengths"""
        return self._regularizers

    @property
    def score(self):
        """Dict containing cross-val scores for all k-folds, and mean and standard deviation of these"""
        return self._cv_score

    @property
    def split_method(self):
        """String of the method used to split the dataset into train/test sets"""
        return self._split_method
    
    @property
    def train_figure(self):
        """For the most recent kfold, (fig, axes) showing training progress"""
        return self._train_figure
        
    @property
    def split_figure(self):
        """(fig, axes) of train/test splitting diagnostic figure"""
        return self._split_figure
    
    @property
    def diagnostics(self):
        """Dict containing diagnostics of the cross-validation loop across all kfolds: models, regularizers, loss values"""
        return self._diagnostics


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
    k : int, default=5
        Number of k-folds (partitions) of `dataset`
    seed : int, default=None
        Seed for PyTorch random number generator used to shuffle data before
        splitting
    channel : int, default=0
        Channel of the dataset to use in determining the splits

    Notes
    -----
    Once initialized, iterate through the datasets like:
        >>> split_iterator = crossval.RandomCellSplitGridded(dataset, k)
        >>> for (train, test) in split_iterator: # iterate through `k` datasets
        >>> ... # working with the n-th slice of `k` datasets
        >>> ... # do operations with train dataset
        >>> ... # do operations with test dataset

    Treats `dataset` as a single-channel object with all data in `channel`.

    The splitting doesn't select (preserve) Hermitian pairs of visibilities.

    All train splits have the highest 1% of cells by gridded weight
    """

    def __init__(self, dataset, k=5, seed=None, channel=0):
        self.dataset = dataset
        self.k = k
        self.channel = channel

        # get indices for cells in the top 1% of gridded weight
        # (we'll want all training sets to have these high SNR points)
        nvis = len(self.dataset.vis_indexed)
        nn = int(nvis * 0.01)
        # get the nn-th largest value in weight_indexed
        w_thresh = np.partition(self.dataset.weight_indexed, -nn)[-nn]
        self._top_nn = torch.argwhere(
            self.dataset.weight_gridded[self.channel] >= w_thresh
        ).T

        # mask these indices
        self.top_mask = torch.ones(
            self.dataset.weight_gridded[self.channel].shape, dtype=bool
        )
        self.top_mask[self._top_nn[0], self._top_nn[1]] = False
        # use unmasked cells that also have data for splits
        self.split_mask = torch.logical_and(
            self.dataset.mask[self.channel], self.top_mask
        )
        split_idx = torch.argwhere(self.split_mask).T

        # shuffle indices to prevent radial/azimuthal patterns in splits
        if seed is not None:
            torch.manual_seed(seed)
        shuffle = torch.randperm(split_idx.shape[1])
        split_idx = split_idx[:, shuffle]

        # split indices into k subsets
        self.splits = torch.tensor_split(split_idx, self.k, dim=1)

    def __iter__(self):
        # current k-slice
        self._n = 0
        return self

    def __next__(self):
        if self._n < self.k:
            test_idx = self.splits[self._n]
            train_idx = torch.cat(
                ([self.splits[x] for x in range(len(self.splits)) if x != self._n]),
                dim=1,
            )
            # add the masked (high SNR) points to the current training set
            train_idx = torch.cat((train_idx, self._top_nn), dim=1)

            train_mask = torch.zeros(
                self.dataset.weight_gridded[self.channel].shape, dtype=bool
            )
            test_mask = torch.zeros(
                self.dataset.weight_gridded[self.channel].shape, dtype=bool
            )
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
        seed (int): (optional) numpy random seed to use for the permutation, for reproducibility

    Once initialized, iterate through the datasets like

    >>> cv = crossval.DartboardSplitGridded(dataset, k)
    >>> for (train, test) in cv: # iterate among k datasets
    >>> ... # working with the n-th slice of k datasets
    >>> ... # do operations with train dataset
    >>> ... # do operations with test dataset

    Notes:
        All train splits have the cells belonging to the shortest dartboard baseline bin.

        The number of points in the splits is in general not equal.
    """

    def __init__(
        self,
        gridded_dataset: GriddedDataset,
        k: int,
        dartboard: Dartboard | None = None,
        seed: int | None = None,
        verbose: bool = True
    ):
        if k <= 0:
            raise ValueError("k must be a positive integer")

        if dartboard is None:
            dartboard = Dartboard(coords=gridded_dataset.coords)

        self.griddedDataset = gridded_dataset
        self.k = k
        self.dartboard = dartboard
        self.verbose = verbose

        # 2D mask for any UV cells that contain visibilities
        # in *any* channel
        stacked_mask = torch.any(self.griddedDataset.mask, dim=0)

        # get qs, phis from dataset and turn into 1D lists
        qs = self.griddedDataset.coords.packed_q_centers_2D[stacked_mask]
        phis = self.griddedDataset.coords.packed_phi_centers_2D[stacked_mask]

        # create the full cell_list
        self.cell_list = self.dartboard.get_nonzero_cell_indices(qs, phis)

        # indices of cells in the smallest q bin that also have data
        small_q_idx = [i for i,l in enumerate(self.cell_list) if l[0] == 0]
        # cells in the smallest q bin
        self.small_q = self.cell_list[:len(small_q_idx)]

        # partition the cell_list into k pieces.
        # first, randomly permute the sequence to make sure
        # we don't get structured radial/azimuthal patterns.
        # also exclude the cells belonging to the smallest q bin from all splits
        # (we'll add these only to the training splits, as they're iterated through)
        if seed is not None:
            np.random.seed(seed)

        self.k_split_cell_list = np.array_split(
            np.random.permutation(self.cell_list[len(small_q_idx):]), k
        )

    @classmethod
    def from_dartboard_properties(
        cls,
        gridded_dataset: GriddedDataset,
        k: int,
        q_edges: NDArray[floating[Any]],
        phi_edges: NDArray[floating[Any]],
        seed: int | None = None,
        verbose: bool = True,
    ) -> DartboardSplitGridded:
        r"""
        Alternative method to initialize a DartboardSplitGridded object from Dartboard parameters.

         Args:
             griddedDataset (:class:`~mpol.datasets.GriddedDataset`): instance of the gridded dataset
             k (int): the number of subpartitions of the dataset
             q_edges (1D numpy array): an array of radial bin edges to set the dartboard cells in :math:`[\mathrm{k}\lambda]`. If ``None``, defaults to 12 log-linearly radial bins stretching from 0 to the :math:`q_\mathrm{max}` represented by ``coords``.
             phi_edges (1D numpy array): an array of azimuthal bin edges to set the dartboard cells in [radians]. If ``None``, defaults to 8 equal-spaced azimuthal bins stretched from :math:`0` to :math:`\pi`.
             seed (int): (optional) numpy random seed to use for the permutation, for reproducibility
             verbose (bool): whether to print notification messages
        """
        dartboard = Dartboard(gridded_dataset.coords, q_edges, phi_edges)
        return cls(gridded_dataset, k, dartboard, seed, verbose)

    def __iter__(self) -> DartboardSplitGridded:
        self.n = 0  # the current k-slice we're on
        return self

    def __next__(self) -> tuple[GriddedDataset, GriddedDataset]:
        if self.n < self.k:
            k_list = self.k_split_cell_list.copy()
            if self.k == 1:
                if self.verbose is True:
                    logging.info("    DartboardSplitGridded: only 1 k-fold: splitting dataset as ~80/20 train/test")
                ntest = round(0.2 * len(k_list[0]))
                # put ~20% of cells into test set
                cell_list_test = k_list[0][:ntest]
                # remove cells in test set from train set
                k_list[0] = np.delete(k_list[0], range(ntest), axis=0)
            else:
                cell_list_test = k_list.pop(self.n)

            # put the remaining indices back into a full list
            cell_list_train = np.concatenate(k_list)
            # add the smallest q bin cells into the train list
            cell_list_train = np.append(cell_list_train, self.small_q, axis=0)

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
