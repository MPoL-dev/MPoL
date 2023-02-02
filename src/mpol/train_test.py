import numpy as np
import torch

from mpol.connectors import GriddedResidualConnector
from mpol.losses import nll_gridded, entropy, sparsity, TV_image, TSV
# from mpol.plot import train_diagnostics # TODO

class TrainTest:
    r"""
    Utilities training and testing an MPoL neural network.

    Parameters
    ----------
    gridder : `mpol.gridding.Gridder` object
        Instance of the `mpol.gridding.Gridder` class.
    optimizer : `torch.optim` object
        PyTorch optimizer class for the training loop.
    config : dict
        Dictionary containing training parameters. 
    verbose : bool, default=True
        Whether to print notification messages. 
    """

    def __init__(self, gridder, optimizer, config, verbose=True):
        self._gridder = gridder
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

