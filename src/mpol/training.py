import logging
import numpy as np
import torch

from mpol.losses import TSV, TV_image, entropy, nll_gridded, sparsity
from mpol.plot import train_diagnostics_fig

class TrainTest:
    r"""
    Utilities for training and testing an MPoL neural network.

    Args:
        imager (:class:`mpol.gridding.DirtyImager` object): Instance of the `mpol.gridding.DirtyImager` class.
        optimizer (:class:`torch.optim` object): PyTorch optimizer class for the training loop.
        epochs (int): Number of training iterations, default=10000
        convergence_tol (float): Tolerance for training iteration stopping criterion as assessed by
            loss function (suggested <= 1e-3)
        regularizers (nested dict): Dictionary of image regularizers to use. For each, a dict of the strength ('lambda', float), whether to guess an initial value for lambda ('guess', bool), and other quantities needed to compute their loss term.
            
            Example:
                ``{"sparsity":{"lambda":1e-3, "guess":False},
                "entropy": {"lambda":1e-3, "guess":True, "prior_intensity":1e-10}
                }``

        train_diag_step (int): Interval at which training diagnostics are output. If None, no diagnostics will be generated.
        kfold (int): The k-fold of the current training set (for diagnostics)        
        save_prefix (str): Prefix (path) used for saved figure names. If None, figures won't be saved
        verbose (bool): Whether to print notification messages
    """

    def __init__(self, imager, optimizer, epochs=10000, convergence_tol=1e-3, 
                regularizers={}, train_diag_step=None, kfold=None, 
                save_prefix=None, verbose=True
                ):
        self._imager = imager
        self._optimizer = optimizer
        self._epochs = epochs
        self._convergence_tol = convergence_tol
        self._regularizers = regularizers
        self._train_diag_step = train_diag_step
        self._save_prefix = save_prefix
        self._kfold = kfold
        self._verbose = verbose

        self._train_figure = None

    def loss_convergence(self, loss):
        r"""
        Estimate whether the loss function has converged by assessing its
        relative change over recent iterations.

        Parameters
        ----------
        loss : array
            Values of loss function over iterations (epochs).
            If len(loss) < 11, `False` will be returned, as convergence
            cannot be adequately assessed.

        Returns
        -------
        `True` if the convergence criterion is met, else `False`.
        """
        min_len = 11
        if len(loss) < min_len:
            return False

        ratios = np.abs(loss[-1] / loss[-min_len:-1])

        return all(1 - self._convergence_tol <= ratios) and all(
            ratios <= 1 + self._convergence_tol
        )


    def loss_lambda_guess(self):
        r"""
        Set an initial guess for regularizer strengths :math:`\lambda_{x}` by
        comparing images generated with different visibility weighting.

        The guesses update `lambda` values in `self._regularizers`.
        """

        # generate images of the data using two briggs robust values
        img1, _ = self._imager.get_dirty_image(weighting='briggs', robust=0.0)
        img2, _ = self._imager.get_dirty_image(weighting='briggs', robust=0.5)
        img1 = torch.from_numpy(img1.copy())
        img2 = torch.from_numpy(img2.copy())

        if self._regularizers.get('entropy', {}).get('guess') == True:
            # force negative pixel values to small positive value
            img1_nn = torch.where(img1 < 0, 1e-10, img1)
            img2_nn = torch.where(img2 < 0, 1e-10, img2)

            loss_e1 = entropy(img1_nn, self._regularizers['entropy']['prior_intensity'])
            loss_e2 = entropy(img2_nn, self._regularizers['entropy']['prior_intensity'])
            guess_e = 1 / (loss_e2 - loss_e1)
            # update stored value
            self._regularizers['entropy']['lambda'] = guess_e.numpy().item()

        if self._regularizers.get('sparsity', {}).get('guess') == True:
            loss_s1 = sparsity(img1)
            loss_s2 = sparsity(img2)
            guess_s = 1 / (loss_s2 - loss_s1)
            self._regularizers['sparsity']['lambda'] = guess_s.numpy().item()

        if self._regularizers.get('TV', {}).get('guess') == True:
            loss_TV1 = TV_image(img1, self._regularizers['TV']['epsilon'])
            loss_TV2 = TV_image(img2, self._regularizers['TV']['epsilon'])
            guess_TV = 1 / (loss_TV2 - loss_TV1)
            self._regularizers['TV']['lambda'] = guess_TV.numpy().item()

        if self._regularizers.get('TSV', {}).get('guess') == True:
            loss_TSV1 = TSV(img1)
            loss_TSV2 = TSV(img2)
            guess_TSV = 1 / (loss_TSV2 - loss_TSV1)
            self._regularizers['TSV']['lambda'] = guess_TSV.numpy().item()


    def loss_eval(self, vis, dataset, sky_cube=None):
        r"""
        Parameters
        ----------
        vis : torch.complex tensor
            Model visibility cube (see `mpol.fourier.FourierCube.forward`)
        dataset : dataset object
            Instance of the `mpol.datasets.GriddedDataset` class.
        sky_cube : torch.double
            MPoL Ground Cube (see `mpol.utils.packed_cube_to_ground_cube`)

        Returns
        -------
        loss : torch.double
            Value of loss function
        """
        # negative log-likelihood loss function
        loss = nll_gridded(vis, dataset)

        # regularizers
        if sky_cube is not None:
            if self._regularizers.get('entropy', {}).get('lambda') is not None:
                loss += self._regularizers['entropy']['lambda'] * entropy(
                    sky_cube, self._regularizers['entropy']['prior_intensity'])
            if self._regularizers.get('sparsity', {}).get('lambda') is not None:
                loss += self._regularizers['sparsity']['lambda'] * sparsity(sky_cube)
            if self._regularizers.get('TV', {}).get('lambda') is not None:
                loss += self._regularizers['TV']['lambda'] * TV_image(
                    sky_cube, self._regularizers['TV']['epsilon'])
            if self._regularizers.get('TSV', {}).get('lambda') is not None:
                loss += self._regularizers['TSV']['lambda'] * TSV(sky_cube)

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
            Value of loss function at end of optimization loop
        losses : list of float
            Loss value at each iteration (epoch) in the loop
        """
        # set model to training mode
        model.train()

        count = 0
        losses = []
        self._train_state = {}

        # guess initial strengths for regularizers in `self._regularizers`
        # that have 'guess':True
        # (this updates `self._regularizers`) 
        self.loss_lambda_guess()

        if self._verbose:
            logging.info("    Image regularizers: {}".format(self._regularizers))

        while not self.loss_convergence(np.array(losses)) and count <= self._epochs:
            if self._verbose:
                logging.info(
                    "\r  Training: epoch {} of {}".format(count, self._epochs),
                    end="",
                    flush=True,
                )

            # check early-on whether the loss isn't evolving
            if count == 20:
                loss_arr = np.array(losses)
                if all(0.9 <= loss_arr[:-1] / loss_arr[1:]) and all(
                    loss_arr[:-1] / loss_arr[1:] <= 1.1
                ):
                    warn_msg = (
                        "The loss function is negligibly evolving. loss_rate "
                        + "may be too low."
                    )
                    logging.info(warn_msg)
                    raise Warning(warn_msg)

            self._optimizer.zero_grad()

            # calculate model visibility cube (corresponding to current pixel
            # values of mpol.images.BaseCube)
            vis = model()

            # get predicted sky cube corresponding to model visibilities
            sky_cube = model.icube.sky_cube

            # calculate loss between model visibilities and data
            loss = self.loss_eval(vis, dataset, sky_cube)
            losses.append(loss.item())

            # calculate gradients of loss function w.r.t. model parameters
            loss.backward()

            # update model parameters via gradient descent
            self._optimizer.step()

            # store current training parameter values
            # TODO: store hyperpar values, access in crossval.py
            self._train_state["kfold"] = self._kfold 
            self._train_state["epoch"] = count
            self._train_state["learn_rate"] = self._optimizer.state_dict()['param_groups'][0]['lr']            

            # generate optional fit diagnostics
            if self._train_diag_step is not None and (count % self._train_diag_step == 0 or count == self._epochs or self.loss_convergence(np.array(losses))):
                train_fig, train_axes = train_diagnostics_fig(
                    model, losses=losses, train_state=self._train_state, 
                    save_prefix=self._save_prefix
                    )
                self._train_figure = (train_fig, train_axes)

            count += 1

        if self._verbose:
            if count < self._epochs:
                logging.info("\n    Loss function convergence criterion met at epoch "
                                "{}".format(count-1))
            else:
                logging.info("\n    Loss function convergence criterion not met; "
                                "training stopped at specified maximum epochs, {}".format(self._epochs))

        # return loss value    
        return loss.item(), losses


    def test(self, model, dataset):
        r"""
        Test a model visibility cube against withheld data.

        Parameters
        ----------
        model : `torch.nn.Module` object
            A neural network; instance of the `mpol.precomposed.SimpleNet` class.
        dataset : PyTorch dataset object
            Instance of the `mpol.datasets.GriddedDataset` class.

        Returns
        -------
        loss.item() : float
            Value of loss function
        """
        # evaluate trained model against a set of withheld (test) visibilities
        model.eval()

        # calculate model visibility cube
        vis = model()

        # calculate loss used for a cross-validation score
        loss = self.loss_eval(vis, dataset)

        # return loss value
        return loss.item()

    @property
    def regularizers(self):
        """Dict containing regularizers used and their strengths"""
        return self._regularizers

    @property
    def train_figure(self):
        """(fig, axes) of figure showing training diagnostics"""
        return self._train_figure
    
    @property
    def train_state(self):
        """Dict containing parameters of interest in the training loop"""
        return self._train_state