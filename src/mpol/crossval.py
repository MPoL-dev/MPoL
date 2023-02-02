import numpy as np
import torch

from mpol.precomposed import SimpleNet
from mpol.datasets import Dartboard, KFoldCrossValidatorGridded
from mpol.train_test import TrainTest

class CrossValidate:
    r"""
    # TODO
    """
    def __init__(self, coords, gridder, config, verbose=True):
        self._coords = coords
        self._gridder = gridder
        self._config = config
        self._verbose = verbose


    def split_dataset(self, dataset):
        r"""
        # TODO
        """
        # create a radial and azimuthal partition for the dataset
        dartboard = Dartboard(coords=self._coords)

        # use 'dartboard' to split full dataset into train/test subsets
        subsets = KFoldCrossValidatorGridded(dataset, self._config["kfolds"],
                                        dartboard=dartboard,
                                        npseed=self._config["seed"])#, device=device) # TODO

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
            optimizer = torch.optim.Adam(model.parameters(), lr=self._config["learn_rate"])

            trainer = TrainTest(self._gridder, optimizer, self._config)
            _, loss_history = trainer.train(model, train_subset)
            loss_histories.append(loss_history)
            all_scores.append(trainer.test(model, test_subset))

        # average individual test scores as a cross-val metric for chosen 
        # hyperparameters
        cv_score = np.mean(all_scores)

        return cv_score, all_scores, loss_histories
    