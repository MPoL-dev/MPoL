import numpy as np
import torch

from mpol.precomposed import SimpleNet
from mpol.datasets import Dartboard, KFoldCrossValidatorGridded
from mpol.train_test import TrainTest

class CrossValidate:
    r"""
    # TODO
    """
    def __init__(self, coords, kfolds, seed, learn_rate, gridder, epochs, 
                convergence_tol, 
                lambda_guess_regularizers, lambda_entropy, 
                entropy_prior_intensity, lambda_sparsity, lambda_TV, 
                TV_epsilon, lambda_TSV, 
                train_diag_step, diag_fig_train, verbose=True):
        self._coords = coords
        self._kfolds = kfolds
        self._seed = seed
        self._learn_rate = learn_rate
        self._gridder = gridder
        self._epochs = epochs
        self._convergence_tol = convergence_tol
        self._lambda_guess_regularizers = lambda_guess_regularizers
        self._lambda_entropy = lambda_entropy
        self._entropy_prior_intensity = entropy_prior_intensity
        self._lambda_sparsity = lambda_sparsity
        self._lambda_TV = lambda_TV
        self._TV_epsilon = TV_epsilon
        self._lambda_TSV = lambda_TSV
        self._train_diag_step = train_diag_step
        self._diag_fig_train = diag_fig_train
        self._verbose = verbose


    def split_dataset(self, dataset, kfolds, seed):
        r"""
        # TODO
        """
        # create a radial and azimuthal partition for the dataset
        dartboard = Dartboard(coords=self._coords)

        # use 'dartboard' to split full dataset into train/test subsets
        subsets = KFoldCrossValidatorGridded(dataset, kfolds,
                                        dartboard=dartboard,
                                        npseed=seed)#, device=device) # TODO

        # store the individual train/test subsets
        test_train_datasets = [(train_pair, test_pair) for (train_pair, test_pair) in subsets]
        # test_train_datasets = [(train.to('cuda'), test.to('cuda')) for (train, test) in subsets] # TODO

        return test_train_datasets


    def run_crossval(self, test_train_datasets):
        r"""
        # TODO
        """
        loss_histories = []
        all_scores = []

        for kfold, (train_subset, test_subset) in enumerate(test_train_datasets):
            print('\nk_fold {} of {}'.format(kfold, np.shape(test_train_datasets)[0] - 1))

            # create a new model and optimizer for this k_fold
            model = SimpleNet(coords=self._coords, nchan=train_subset.nchan)
            # if use_gpu: # TODO
                # model = model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=self._learn_rate)

            trainer = TrainTest(self._gridder, optimizer, self._epochs, 
                                self._convergence_tol, 
                                self._lambda_guess_regularizers, 
                                self._lambda_entropy,
                                self._entropy_prior_intensity,
                                self._lambda_sparsity,
                                self._lambda_TV, self._TV_epsilon,
                                self._lambda_TSV,
                                self._train_diag_step, self._diag_fig_train,
                                self._verbose
            )

            _, loss_history = trainer.train(model, train_subset)
            loss_histories.append(loss_history)
            all_scores.append(trainer.test(model, test_subset))

        # average individual test scores as a cross-val metric for chosen 
        # hyperparameters
        cv_score = np.mean(all_scores)

        return cv_score, all_scores, loss_histories
    