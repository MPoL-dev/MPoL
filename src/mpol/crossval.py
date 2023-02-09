import numpy as np
import logging
import torch

from mpol.precomposed import SimpleNet
from mpol.datasets import Dartboard, KFoldCrossValidatorGridded
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
    def __init__(self, coords, gridder, kfolds=5, seed=None, learn_rate=0.5, 
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
        kfolds : int 
            Number of k-folds to use in cross-validation
        seed : int 
            Seed for random number generator used in splitting data

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

        if self._split_diag_fig:
            splitter_diagnostics_fig(split_iterator, save_prefix=self._save_prefix)
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
    