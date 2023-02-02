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

