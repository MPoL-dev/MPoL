=====
Usage
=====

MPoL is a Regularized Maximum Likelihood (RML) imaging package built on top of the machine learning framework `PyTorch <https://pytorch.org/>`_. The key ingredient that MPoL provides is the :class:`mpol.images.ImageCube` module. This is a PyTorch layer that links the base parameter set (the image cube pixels) to the dataset (the complex visibilities) through the FFT and band-limited interpolation routines used in the radio astronomy community. The MPoL package is designed such that you use the native infrastructure of PyTorch to write custom optimization routines to interact with :class:`mpol.images.ImageCube`. You don't need to know PyTorch to use MPoL, but it doesn't hurt to spend a little time browsing the `tutorials <https://pytorch.org/tutorials/>`_.

This document will give you an idea of one workflow to generate an image from a set of visibilities, and in the process demonstrate some of the core functionality of the package. 

Typically, you'll want to import the following depedencies ::

    import numpy as np
    import torch

    import mpol.gridding
    from mpol.losses import loss_fn, loss_fn_entropy
    from mpol.images import Image
    from mpol.constants import *


Data
----

The fundamental dataset is the set of complex-valued visibility measurements: lists of the :math:`u`, :math:`v` coordinates, real and imaginary visibility values (:math:`\Re` and :math:`\Im`), and visibility weights (:math:`w \propto \frac{1}{\sigma^2}`). If you have a CASA measurement set, you'll want to export these quantities. You can achieve this by writing a CASA script yourself or using the `UVHDF5 package <https://github.com/AstroChem/UVHDF5>`_. It goes without saying that you should make sure the data are correctly calibrated, particularly the `weights <https://casaguides.nrao.edu/index.php/DataWeightsAndCombination>`_. ::

    uu = # your data here in [kilolam] 
    vv = # your data here in [kilolam]
    data_re = # your data here in [Jy]
    data_im = # your data here in [Jy]
    weights = # your data here in [1/Jy^2]

For convenience, we provide a dataset wrapper for these quantities, :class:`mpol.datasets.UVDataset`. 


Image Model 
-----------

You will need to decide how many pixels (``npix``) to use in your image and how large each pixel will be (``cell_size``, in arcseconds). You will want to make an image that is large enough to contain all of the emission in the dataset, because otherwise you will alias bright sources into your image. You will also want to make ``cell_size`` small enough so that you can capture the highest spatial frequency visibility sampled by your dataset. 


:class:`mpol.images.Image` is a simplified version of :class:`mpol.images.ImageCube`, which assumes there is only one frequency/velocity channel to the dataset. There are different instantiation options depending on whether and how you would like to pre-grid the dataset, so be sure to check out the API documentation on these objects. Pre-gridding the dataset is almost always the faster option, but removes the option for some more fine-grained optimization schemes. ::

    model = Image(
        cell_size=10.0 / npix,
        npix=npix,
        image=None,
        dataset=dataset,
        grid_indices=grid_indices,
    )

The main functionility of the Image class is to forward-model the visibilities starting from an image, done using the ``Image.forward`` method. This method is called automatically when you use ``model()``. To save computation, the core image representation is actually stored `pre-fftshifted <https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftshift.html>`_ in the ``model._image`` variable, but you can query the de-shifted version using `model.image`. 

Since we have just initialized the model, we can see that `model.image` is blank. If you have a better starting image, you can pass this as a PyTorch tensor to the ``image`` parameter.

We've also chosen to grid the data, with ``grid=True``.


Optimizer 
---------

Define an optimizer ::

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


Losses
------

In the parlance of the machine learning community, one can define loss functions against the model image and visibilities. For regularized maximum likelihood imaging, one key loss function that we are interested in is the data likelihood. 

:func:`mpol.losses.loss_fn`


Training loop 
-------------

This is to train. Choose some range to iterate over. It's a good idea to track the loss, as well. ::

    for i in range(1000):
        # clears the gradients of all optimized tensors
        optimizer.zero_grad()

        # query the model for the new model visibilities
        model_vis = model()

        # calculate the loss function
        loss_nll = loss_fn(model_vis, (g_re, g_im, g_weights))
        loss_entropy = 0.1 * loss_fn_entropy(model.image, prior_intensity=0.05)

        # needs to be fftshifted
        loss_TV = 0.5 * loss_fn_TV(model.image)
        loss = loss_nll + loss_entropy + loss_TV

        l_nll.append(loss_nll.item())
        l_entropy.append(loss_entropy.item())

        # compute the intermediate gradients that go into
        # calculating the loss and attach them to the image
        loss.backward()

        optimizer.step()

        # clip negative image values to positive, if using the entropy term
        model._image.data[model._image.data < 0] = 1e-10


Saving output 
-------------

You can save the output ::

        img = model.image.detach().numpy()
        np.save("image.npy", img)

