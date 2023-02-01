import numpy as np
import torch

class TrainTest:
    r"""
    Utilities training and testing an MPoL neural network.

    Parameters
    ----------
    model : `torch.nn.Module` class 
        A neural network. Instance of the `mpol.precomposed.SimpleNet` class.
    dataset : PyTorch dataset object
        Instance of the `mpol.datasets.GriddedDataset` class.
    optimizer : `torch.optim` class
        PyTorch optimizer class for the training loop.
    config : dict
        Dictionary containing training parameters. 
    verbose : bool, default=True
        Whether to print notification messages. 
    """

    def __init__(self, model, dataset, optimizer, config, verbose=True):
        self._model = model
        self._dataset = dataset
        self._optimizer = optimizer
        self._config = config
        self._verbose = verbose
