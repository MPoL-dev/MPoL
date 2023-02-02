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
