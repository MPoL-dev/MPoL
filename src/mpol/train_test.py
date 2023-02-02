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

        
    def loss_lambda_guess(self):
        r"""
        Set an initial guess for regularizer strengths :math:`\lambda_{x}` by 
        comparing images generated with different visibility weighting. 
        
        The guesses update \lambda values in the `self._config` dictionary.
        """
        # generate images of the data using two briggs robust values
        img1, _ = self._gridder.get_dirty_image(weighting='briggs', robust=0.0)
        img2, _ = self._gridder.get_dirty_image(weighting='briggs', robust=0.5)
        img1 = torch.from_numpy(img1.copy())
        img2 = torch.from_numpy(img2.copy())

        if "entropy" in config["lambda_guess_regularizers"]:
            loss_e1 = entropy(img1_nn, config["entropy_prior_intensity"])
            loss_e2 = entropy(img2_nn, config["entropy_prior_intensity"])
            # update config value
            self._config["lambda_entropy"] = 1 / (loss_e2 - loss_e1)

        if "sparsity" in config["lambda_guess_regularizers"]:
            loss_s1 = sparsity(img1_nn)
            loss_s2 = sparsity(img2_nn)
            self._config["lambda_sparsity"] = 1 / (loss_s2 - loss_s1)

        if "TV" in config["lambda_guess_regularizers"]:
            loss_TV1 = TV_image(img1_nn, config["TV_epsilon"])
            loss_TV2 = TV_image(img2_nn, config["TV_epsilon"])
            self._config["lambda_TV"] = 1 / (loss_TV2 - loss_TV1)

        if "TSV" in config["lambda_guess_regularizers"]:
            loss_TSV1 = TSV(img1_nn)
            loss_TSV2 = TSV(img2_nn)
            self._config["lambda_TSV"] = 1 / (loss_TSV2 - loss_TSV1)


    def loss_eval(self, vis, dataset, sky_cube=None):
        r"""
        Parameters
        ----------
        vis : torch.complex tensor
            Model visibility cube (see `mpol.fourier.FourierCube.forward`)
        dataset : PyTorch dataset object
            Instance of the `mpol.datasets.GriddedDataset` class.
        sky_cube : torch.double
            MPoL Ground Cube (see `mpol.utils.packed_cube_to_ground_cube`)

        Returns
        -------
        loss : torch.double
            Loss value

        """
        # negative log-likelihood loss function
        loss = nll_gridded(vis, dataset)

        # regularizers
        if sky_cube is not None:
            # optionally guess regularizer strengths
            if self._config["lambda_guess_regularizers"] is not None:
                self.loss_lambda_guess()

            # apply regularizers
            if self._config["lambda_entropy"] is not None:
                loss += self._config["lambda_entropy"] * entropy(sky_cube, 
                                                                self._config["entropy_prior_intensity"])
            if self._config["lambda_sparsity"] is not None:
                loss += self._config["lambda_sparsity"] * sparsity(sky_cube)
            if self._config["lambda_TV"] is not None:
                loss += self._config["lambda_TV"] * TV_image(sky_cube, 
                                                            self._config["TV_epsilon"])
            if self._config["lambda_TSV"] is not None:
                loss += self._config["lambda_TSV"] * TSV(sky_cube)

        return loss 


    def train(self, model, dataset):
        r"""
        Trains a neural network, forward modeling a visibility dataset and 
        evaluating the corresponding model image against the data, using 
        PyTorch with gradient descent.
        
        Parameters
        ----------
        model : `torch.nn.Module` object
            A neural network; instance of the `mpol.precomposed.SimpleNet` class.
        dataset : PyTorch dataset object
            Instance of the `mpol.datasets.GriddedDataset` class.

        Returns
        -------
        loss.item() : float
            Loss value 
        """
        # set model to training mode
        model.train()
        
        # track model residuals
        residuals = GriddedResidualConnector(model.fcube, dataset)

        count = 0
        # track loss value over epochs
        losses = []
        
        while (not self.loss_convergence(np.array(losses),
                                        self._config["convergence_tol"])
                and count <= self._config["epochs"]):

            if self._verbose:
                print('\r  epoch {} of {}'.format(count, 
                                                 self._config["epochs"]), 
                        end='', flush=True)
            
            self._optimizer.zero_grad()

            # calculate model visibility cube (corresponding to current pixel 
            # values of mpol.images.BaseCube)
            vis = model.forward() 

            # get predicted sky cube corresponding to model visibilities
            sky_cube = model.icube.sky_cube

            # calculate loss between model visibilities and data
            loss = self.loss_eval(vis, dataset, sky_cube)
            losses.append(loss.item())

            # generate optional fit diagnostics
            if (count % self._config["learn_diag_step"] == 0 or
                count == self._config["epochs"] - 1) :

                if self._config["diag_fig_train"]:
                    train_diagnostics(model, residuals, losses, count)

            # calculate gradients of loss function w.r.t. model parameters
            loss.backward() 

            # update model parameters via gradient descent
            self._optimizer.step()  

            count += 1

        # return loss value    
        return loss.item(), losses


