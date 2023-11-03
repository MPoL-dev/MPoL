.. _gpu-reference-label:

GPU Acceleration
----------------

Installation and Configuration
==============================

Installing CUDA Toolkit
~~~~~~~~~~~~~~~~~~~~~~~

The first step in utilizing the computational power of your Nvidia GPU
is to install the CUDA toolkit. (If you've already configured your GPU for other software, you may skip this step.) To download the installer, visit `this
link <https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_network>`__
and provide your system info to download the installer. You must be
using either Linux or Windows, and you must be using one of
`these <https://developer.nvidia.com/cuda-gpus>`__ graphics cards. Once
the toolkit has been installed, follow the instructions in the installer
GUI. Once complete, restart your computer.

Python and PyTorch GPU configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next step is to check whether your Python and PyTorch installations are correctly configured to use your GPU. After installing MPoL, you can check whether everything installed correctly by opening up a Python interpreter, ard running

.. code:: ipython3

    import torch
    print(torch.cuda.is_available())

This command should return ``True``. If not, then you may need to use a more specific installation process. Go to the `PyTorch Official Site <https://pytorch.org/>`__ and scroll down
on the page until you see the **Install PyTorch** section. Input your
specifications for your needs into this area and use the text that is
generated for your install. For example, making of this tutorial on a Windows
10 system with a Nvidia GTX 1080 required specific pip installation,
while another Windows 10 system using a Nvidia GTX 1660Ti worked with the default
``pip install torch torchvision``. Your mileage may vary.

Why use the GPU?
================

Using a GPU can accelerate computing speeds up to 100x over CPUs, especially for operations on large images, like is common for MPoL. The following is a quick example showing the addition of two large vectors. Your exact timing may vary, but for our hardware this calculation took
320 milliseconds seconds on the CPU, while it only took 3.1 milliseconds on the GPU.

.. code:: ipython3

    import torch
    import time
    N = int(9.9e7)
    A = torch.ones(N)
    B = torch.ones(N)
    start = time.time()
    C = A + B
    print(time.time() - start)

.. code:: ipython3

    torch.cuda.empty_cache() # emptying the cache on the gpu just incase there was any memory left over from an old operation
    A = A.cuda()
    B = B.cuda()

.. code:: ipython3

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    C = A + B
    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

Using the GPU as part of PyTorch and MPoL
=========================================

Here is a short example demonstrating how to initialize an MPoL model and run it on the GPU. First we will set our device to the CUDA device.

.. code:: ipython3

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)


.. parsed-literal::

    cuda:0


This if-else statement is used just to ensure that we aren’t trying to
run PyTorch on the GPU if it isn’t available. The rest of this tutorial
will assume that ``device=cuda:0``.

.. note::
    ``cuda:0`` is technically only required if you have more than one GPU. ``device='cuda'`` will instruct PyTorch to use the default cuda device.

Now that we have our device set, we'll initialize the MPoL dataset as in previous tutorials. This example uses a multi-channel dataset, but for demonstration purposes we will only use the central
channel (``central_chan=4``).

.. code:: ipython3

    from astropy.utils.data import download_file
    import numpy as np
    from mpol import gridding, coordinates
    fname = download_file(
        'https://zenodo.org/record/4498439/files/logo_cube.npz',
        cache=True,
        )
    d = np.load(fname)
    coords = coordinates.GridCoords(cell_size=0.03, npix=180)
    central_chan = 4
    gridder = gridding.Gridder(
        coords=coords,
        uu=d['uu'][central_chan],
        vv=d['vv'][central_chan],
        weight=d['weight'][central_chan],
        data_re=d['data_re'][central_chan],
        data_im=d['data_im'][central_chan],
    )
    dataset = gridder.to_pytorch_dataset()

Next we'll create a :class:`~mpol.precomposed.SimpleNet` module to train to our
data. For more detailed
information, see the `Optimization
Loop <optimization.html>`__
tutorial or the MPoL SimpleNet `Source
Code <https://mpol-dev.github.io/MPoL/_modules/mpol/precomposed.html#SimpleNet>`__.

.. code:: ipython3

    from mpol.precomposed import SimpleNet
    model = SimpleNet(coords=coords, nchan=dataset.nchan)

We are now ready to move our model and data to the GPU using the ``tensor.to(device)``
functionality common to most PyTorch objects. One can
also use the ``tensor.cuda()`` to move the tensor to the default CUDA
device. Both of these methods return a *copy* of the object on the GPU.

We've borrowed a ``config`` dictionary from the `Cross Validation
Tutorial <crossvalidation.html>`__, which basically contains a set of parameters that resulted in a strong cross validation score for this particular dataset. For more
details on these variables, see the `Cross Validation
Tutorial <crossvalidation.html>`__.

.. code:: ipython3

    dset = dataset.to(device)
    model = model.cuda()
    config = {'lr':0.5, 'lambda_sparsity':1e-4, 'lambda_TV':1e-4, 'epochs':600}
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

We are now ready to train our network on the GPU. We will use a for-loop
with 600 iterations (epochs) in which we will calculate the loss and
step our optimizer.

.. code:: ipython3

    from mpol import losses

    # set the model to training mode
    model.train()
    for i in range(config['epochs']):
        # set the model to zero grad
        model.zero_grad()

        # forward pass
        vis = model()

        # get skycube from our forward model
        sky_cube = model.icube.sky_cube

        # compute loss
        loss = (
            losses.nll_gridded(vis, dset)
            + config['lambda_sparsity'] * losses.sparsity(sky_cube)
            + config['lambda_TV'] * losses.TV_image(sky_cube))

        # perform a backward pass
        loss.backward()

        # update the weights
        optimizer.step()

Congratulations! You have now trained a neural network on your GPU. In general, the process for running on the GPU is designed to be simple. Once your
CUDA device has been set-up, the main changes to a CPU-only run are the steps requried moving the data and the model to the GPU for training.
