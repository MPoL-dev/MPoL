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


    def loss_convergence(self, loss, tol=1e-2):
        r"""
        Estimate whether the loss function has converged by assessing its 
        relative change over recent iterations.
        
        Parameters
        ----------
        loss : array
            Values of loss function over iterations (epochs). 
            If len(loss) < 11, `False` will be returned, as convergence 
            cannot be adequately assessed.
        tol : float > 0, default=1e-2
            Tolerence for convergence

        Returns
        -------
        `True` if the convergence criterion is met, else `False`.
        """
        min_len = 11 
        if len(loss) < min_len:
            return False
        
        ratios = np.abs(loss[-1] / loss[-min_len:-1]) 

        return np.all(1 - tol <= ratios) and np.all(ratios <= 1 + tol)

        
        return np.all(np.abs(loss_new - loss_old) <= tol * loss_new)

