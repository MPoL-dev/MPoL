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


    def check_convergence(self, loss_new, loss_old, tol):
        r"""
        Determine whether the loss function has converged.
        
        Parameters
        ----------
        loss_new : float
            Current value of loss function 
        loss_old : float
            Previous value of loss function
        tol : float > 0, default = 1e-3
            Tolerence for convergence

        Returns
        -------
        `True` if the convergence criterion is met, else `False`.
        """
        
        return np.all(np.abs(loss_new - loss_old) <= tol * loss_new)

