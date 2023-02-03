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
    epochs : int, default=500
        Number of training iterations
    convergence_tol : float, default=1e-2
        Tolerance for training iteration stopping criterion as assessed by 
        loss function (suggested <= 1e-2)
    lambda_guess : list of str, default=None
        List of regularizers for which to guess an initial value 
    lambda_entropy : float
        Relative strength for entropy regularizer
    entropy_prior_intensity : float, default=1e-10
        Prior value :math:`p` to calculate entropy against (suggested <<1)
    lambda_sparsity : float, default=None 
        Relative strength for sparsity regularizer
    lambda_TV : float, default=None
        Relative strength for total variation (TV) regularizer
    TV_epsilon : float, default=1e-10
        Softening parameter for TV regularizer (suggested <<1)
    lambda_TSV : float, default=None
        Relative strength for total squared variation (TSV) regularizer
    train_diag_step : int, default=None
        Interval at which optional training diagnostics are output
    diag_fig_train : bool, default=False
        Whether to generate a diagnostic figure during training
        (if True, `train_diag_step` must also be nonzero)
    verbose : bool, default=True
        Whether to print notification messages. 
    """

    def __init__(self, gridder, optimizer, epochs=500, convergence_tol=1e-2, 
                lambda_guess=None, lambda_entropy=None, 
                entropy_prior_intensity=1e-10, lambda_sparsity=None, lambda_TV=None, 
                TV_epsilon=1e-10, lambda_TSV=None, 
                train_diag_step=None, diag_fig_train=False, verbose=True):
        self._gridder = gridder
        self._optimizer = optimizer
        self._epochs = epochs
        self._convergence_tol = convergence_tol
        self._lambda_guess = lambda_guess
        self._lambda_entropy = lambda_entropy
        self._entropy_prior_intensity = entropy_prior_intensity
        self._lambda_sparsity = lambda_sparsity
        self._lambda_TV = lambda_TV
        self._TV_epsilon = TV_epsilon
        self._lambda_TSV = lambda_TSV
        self._train_diag_step = train_diag_step
        self._diag_fig_train = diag_fig_train
        self._verbose = verbose


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

        return np.all(1 - self._convergence_tol <= ratios) and np.all(ratios <= 1 + self._convergence_tol)


    def loss_lambda_guess(self):
        r"""
        Set an initial guess for regularizer strengths :math:`\lambda_{x}` by 
        comparing images generated with different visibility weighting. 
        
        The guesses update \lambda values in `self`.
        """
        # generate images of the data using two briggs robust values
        img1, _ = self._gridder.get_dirty_image(weighting='briggs', robust=0.0)
        img2, _ = self._gridder.get_dirty_image(weighting='briggs', robust=0.5)
        img1 = torch.from_numpy(img1.copy())
        img2 = torch.from_numpy(img2.copy())

        if "entropy" in self._lambda_guess:
            # force negative pixel values to small positive value
            img1_nn = torch.where(img1 < 0, 1e-10, img1)
            img2_nn = torch.where(img2 < 0, 1e-10, img2)

            loss_e1 = entropy(img1_nn, self._entropy_prior_intensity)
            loss_e2 = entropy(img2_nn, self._entropy_prior_intensity)
            # update stored value
            self._lambda_entropy = 1 / (loss_e2 - loss_e1)

        if "sparsity" in self._lambda_guess:
            loss_s1 = sparsity(img1)
            loss_s2 = sparsity(img2)
            self._lambda_sparsity = 1 / (loss_s2 - loss_s1)

        if "TV" in self._lambda_guess:
            loss_TV1 = TV_image(img1, self._TV_epsilon)
            loss_TV2 = TV_image(img2, self._TV_epsilon)
            self._lambda_TV = 1 / (loss_TV2 - loss_TV1)

        if "TSV" in self._lambda_guess:
            loss_TSV1 = TSV(img1)
            loss_TSV2 = TSV(img2)
            self._lambda_TSV = 1 / (loss_TSV2 - loss_TSV1)


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
            Value of loss function
        """
        # negative log-likelihood loss function
        loss = nll_gridded(vis, dataset)

        # regularizers
        if sky_cube is not None:
            if self._lambda_entropy is not None:
                loss += self._lambda_entropy * entropy(sky_cube, 
                                                        self._entropy_prior_intensity)
            if self._lambda_sparsity is not None:
                loss += self._lambda_sparsity * sparsity(sky_cube)
            if self._lambda_TV is not None:
                loss += self._lambda_TV * TV_image(sky_cube, 
                                                    self._TV_epsilon)
            if self._lambda_TSV is not None:
                loss += self._lambda_TSV * TSV(sky_cube)

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
        losses : list
            Loss value at each iteration (epoch) in the loop
        """
        # set model to training mode
        model.train()
        
        # track model residuals
        residuals = GriddedResidualConnector(model.fcube, dataset)

        count = 0
        # track loss value over epochs
        losses = []
        
        # optionally guess initial regularizer strengths
        if self._lambda_guess is not None:
            # guess, update lambda values in 'self'
            self.loss_lambda_guess()

        while (not self.loss_convergence(np.array(losses))
                and count <= self._epochs):

            if self._verbose:
                print('\r  epoch {} of {}'.format(count, self._epochs), end='', 
                    flush=True)
            
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
            # TODO: uncomment when plot.train_diagnostics in codebase 
            # if self._train_diag_step is not None and (count % self._train_diag_step == 0 or
            #     count == self._epochs - 1) : 
                # if self._diag_fig_train:
                #     train_diagnostics(model, residuals, losses, count)

            # calculate gradients of loss function w.r.t. model parameters
            loss.backward() 

            # update model parameters via gradient descent
            self._optimizer.step()  

            count += 1

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
        vis = model.forward()

        # calculate loss used for a cross-validation score
        loss = self.loss_eval(vis, dataset)

        # return loss value
        return loss.item()
