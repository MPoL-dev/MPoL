=====
Usage
=====

.. note::
    
    You can try out MPoL in your browser using this example `Google Colaboratory Notebook <https://colab.research.google.com/drive/1CDLlDwIDHzhsqSdzZM112lY2x_L8ETcV>`_. This notebook also includes an example of how to run on a GPU.

MPoL is a Regularized Maximum Likelihood (RML) imaging package built on top of the machine learning framework `PyTorch <https://pytorch.org/>`_. The key ingredient that MPoL provides is the :class:`mpol.images.ImageCube` module. This is a PyTorch layer that links the base parameter set (the image cube pixels) to the dataset (the complex visibilities) through the FFT and band-limited interpolation routines used in the radio astronomy community. The MPoL package is designed such that you use the native infrastructure of PyTorch to write custom optimization routines to interact with :class:`mpol.images.ImageCube`. You don't need to know PyTorch to use MPoL, but it doesn't hurt to spend a little time browsing the `tutorials <https://pytorch.org/tutorials/>`_.

This document will give you an idea of one workflow to generate an image from a set of visibilities, and in the process demonstrate some of the core functionality of the package. 

Typically, you'll want to import the following depedencies ::

    import numpy as np
    import torch

    from mpol.losses import loss_fn, loss_fn_sparsity
    from mpol.images import ImageCube
    from mpol.datasets import UVDataset
    from mpol.constants import *


Data
----

The fundamental dataset is the set of complex-valued visibility measurements: lists of the :math:`u`, :math:`v` coordinates, real and imaginary visibility values (:math:`\Re` and :math:`\Im`), and visibility weights (:math:`w \propto \frac{1}{\sigma^2}`). If you have a CASA measurement set, you'll want to export these quantities. You can achieve this by writing a CASA script yourself or using the `UVHDF5 package <https://github.com/AstroChem/UVHDF5>`_. It goes without saying that you should make sure the data are correctly calibrated, particularly the `weights <https://casaguides.nrao.edu/index.php/DataWeightsAndCombination>`_. 

One important thing to note is that the effective CASA (and AIPS) baseline convention is opposite that of the interferometric coordinate system typically presented in textbooks, e.g., `TMS <https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract>`_, Fig. 3.2. Urvashi Rao has a very `helpful memo <https://casa.nrao.edu/casadocs/casa-5.6.0/memo-series/casa-memos/casa_memo2_coordconvention_rau.pdf>`_ explaining the baseline convention in CASA. Essentially, this means you'll want to take the complex conjugate of your visibilities to have your images show up in the correct orientation. ::

    uu = # your data here in [kilolam] 
    vv = # your data here in [kilolam]
    data_re = # your data here in [Jy]
    # perform the complex conjugate by multiplying the imaginaries by -1
    data_im = # -1.0 * (your data here) in [Jy]
    weights = # your data here in [1/Jy^2]

To test out the package, you can play with a mock dataset of Saturn available `here <https://zenodo.org/record/3634225#.XjeyDBNKiL8>`_::

    npzfile = np.load("data.npz")
    uu = npzfile["uu"] # [kilolambda]
    vv = npzfile["vv"] # [kilolambda]
    data_re = npzfile["re"] # [Jy]
    # perform the complex conjugate by multiplying the imaginaries by -1
    data_im = -1.0 * npzfile["im"] # [Jy]
    weights = npzfile["weights"] # [1/Jy]

For convenience, we provide a dataset wrapper for these quantities, :class:`mpol.datasets.UVDataset`. After loading your data, you can initialize this with ::

    dataset = UVDataset(uu=uu, vv=vv, data_re=data_re, data_im=data_im, weights=weights)

However, if we already know the image dimensions that we would like to use, the optimization loop can be greatly sped up if we pre-grid the dataset to the RFFT output grid. You can do this by providing both of the ``cell_size`` and ``npix`` optional keywords to ``UVDataset``. If you don't know apriori how big your source is on the sky, it's always a good idea to make as large an image as possible. Otherwise, if you make a very small image, you will alias emission back into your map. To save you some time, the dataset was made with a (512x512) image of Saturn scaled to 8 arcseconds wide (this is actually smaller than it appears from Earth), so anything larger and more finely gridded than this should be fine ::

    # pre-grid visibilities to anticipated output RFFT grid
    npix = 800
    dataset = UVDataset(uu=uu, vv=vv, data_re=data_re, data_im=data_im, weights=weights, cell_size=8.0/npix, npix=npix)

Image Model 
-----------

If you didn't already set them in the ``UVDataset`` stage, you will need to decide how many pixels (``npix``) to use in your image and how large each pixel will be (``cell_size``, in arcseconds). You will want to make an image that is large enough to contain all of the emission in the dataset, because otherwise you will alias bright sources into your image. You will also want to make ``cell_size`` small enough so that you can capture the highest spatial frequency visibility sampled by your dataset. 


The :class:`mpol.images.ImageCube` requires the following options ::

    model = ImageCube(
        cell_size=8.0 / npix, npix=npix, cube=None, nchan=1
    )


We set the number of channels to 1, since we just have a single channel map. 

The main functionility of the Image class is to forward-model the visibilities starting from an image, done using the ``Image.forward`` method. This method is called automatically when you use ``model()``. To save computation, the core image representation is actually stored `pre-fftshifted <https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftshift.html>`_ in the ``model._cube`` variable, but you can query the de-shifted version using ``model.cube``. 

Since we have just initialized the model, we can see that ``model.cube`` is blank. If you have a better starting image, you can pass this as a PyTorch tensor to the ``cube`` parameter.

Optimizer 
---------

Define an optimizer ::

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

As we'll see in a moment, this optimizer will advance the parameters (in this case, the pixel values of the image cube) based upon the gradient of the loss function with respect to those parameters. PyTorch has many different `optimizers <https://pytorch.org/docs/stable/optim.html#module-torch.optim>`_ available, and it would be worthwhile to try out some of the different ones. Stochastic Gradient Descent (SGD) is one of the simplest, so we'll start here. The ``lr`` parameter is the 'loss rate,' or how ambitious the optimizer should be in taking descent steps. Tuning this requires a bit of trial and error: you want the loss rate to be small enough so that the algorithm doesn't diverge but large enough so that the optimization completes in a reasonable amount of time. 

Losses
------

In the parlance of the machine learning community, one can define loss functions against the model image and visibilities. For regularized maximum likelihood imaging, one key loss function that we are interested in is the data likelihood (:func:`mpol.losses.loss_fn`), which is just the :math:`\chi^2` of the visibilities. Because imaging is an ill-defined inverse problem, however, the visibility likelihood function is not sufficient. We also need to apply regularization to narrow the set of possible images towards ones that we believe are more realistic. The :mod:`mpol.losses` module contains several loss functions currently popular in the literature, so you can experiment to see which best suits your application.


Training loop 
-------------

Next, we'll set up a loop that will 

    1) evaluate the current ``model`` (i.e., the image cube) against the loss functions
    2) calculate the gradients of the loss w.r.t. the model 
    3) advance the ``model`` so as to minimize the loss 

Here is a minimal loop that will accomplish this and track the value of the loss with each iteration. ::

    loss_log = []

    for i in range(1000):
        # clears the gradients of all optimized tensors
        optimizer.zero_grad()

        # query the model for the new model visibilities
        model_vis = model(dataset)

        # calculate the losses
        loss_nll = loss_fn(model_vis, (dataset.re, dataset.im, dataset.weights))
        loss_sparse = 0.1 * loss_fn_sparsity(model.cube)

        loss = loss_nll + loss_sparse
        loss_log.append(loss.item())

        # compute the intermediate gradients that go into
        # calculating the loss and attach them to the image
        loss.backward()

        # advance the optimizer
        optimizer.step()

        # you can also query the current cube value as `model.cube`

It is an excellent idea to track and plot diagnostics like the loss values while optimizing. This will help gain intuition for how the penalty terms (the scale factor in front of the sparsity regularization) affect the image quality. You can also query and save the image cube values and RFFT output during optimization as well.

Moreover, you can compose many intricate optimization strategies using the tools available in PyTorch.

Saving output 
-------------

When you are finished optimizing, you can save the output ::

    cube = model.cube.detach().numpy()
    np.save("cube.npy", cube)

Image bounds for ``matplotlib.pyplot.imshow`` are available in ``model.extent``.

