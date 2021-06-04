Part 1
======

Installing CUDA Toolkit
~~~~~~~~~~~~~~~~~~~~~~~

The first step in utilizing the computational power of your Nvidia GPU
is to install the CUDA toolkit. This can be done by visiting `this
link <https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_network>`__
and inputting your system info to download the installer. You must be
using either Linux or Windows, and you must be using one of
`these <https://developer.nvidia.com/cuda-gpus>`__ graphics cards. Once
the toolkit has been installed, follow the instructions in the installer
GUI. Once complete, restart your computer.

Compatibility With PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~

For MPoL purposes, we want to utilize the full extent of PyTorch
capabilities on the GPU. To do this, open up a python-enabled terminal.
Here we will use the Anaconda Prompt and pip to install the
dependencies. Type in ``pip install torch torchvision`` and let the
installation run. At this point, we now must check if everything
installed correctly and PyTorch can see your now CUDA-enabled device.
One can do this by running ``torch.cuda.is_available()`` as seen below:

.. code:: ipython3

    import torch
    print(torch.cuda.is_available())


.. parsed-literal::

    True
    

Potential Issues
~~~~~~~~~~~~~~~~

If you complete this process and ``torch.cuda.is_available()`` returns
False, then you may need to use a more specific installation process. Go
to the `PyTorch Official Site <https://pytorch.org/>`__ and scroll down
on the page until you see the **Install PyTorch** section. Input your
specifications for your needs into this area and use the text that is
generated for your install. In the making of this tutorial, one Windows
10 system using a Nvidia GTX 1080 required
``pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html``
while another Windows 10 system using a Nvidia GTX 1660Ti worked with
``pip install torch torchvision``. It is unclear why one system may
require one installation command over another.

Why is This Important?
~~~~~~~~~~~~~~~~~~~~~~

Using the GPU over the CPU can show up to a 50x to 100x increase in
computing speeds for large volumes of data. Since MPoL works with
images, the switch to the GPU for training Neural Networks causes a
significant boost to training speed. For a quick example on this,
consult the code below. For a specific run, the time decreased from
0.133 seconds on the CPU to 0.0349 seconds on a GPU for this simple
operation. *Note: timing will change run to run, but in general running
operations on the GPU, especially for a large amount of operations, is
faster*

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

    start = time.time()
    C = A + B
    print(time.time() - start)

Part 2
======

CUDA for Optimization
~~~~~~~~~~~~~~~~~~~~~

Here we will utilize PyTorch’s ability to run on the GPU to give a short
example on initializing a model and running it on the GPU. We will walk
through the steps to create tensors and neural networks and transfer
them to the GPU for the acceleration benefits described in **Part 1**.

First we will set our device to the CUDA device.

.. code:: ipython3

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)


.. parsed-literal::

    cuda:0
    

This if-else statement is used just to ensure that we aren’t trying to
run PyTorch on the GPU if it isn’t available. The rest of this tutorial
will assume that ``device=cuda:0``. *Note: ‘cuda:0’ is technically only
required if you have more than one GPU, if ``device='cuda'`` then
PyTorch will use the default cuda device.*

Now that we have our device set, let us create some data-filled tensor
objects from the mock ALMA dataset. This is a multi-channel dataset
which is represented as a data cube. Here we will use the central
channel of the cube for demonstration purposes, this corresponds to
``central_chan=4``.

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

Now let us create a SimpleNet Neural Network that we will train with our
data. A SimpleNet, as defined by MPoL, is a combination of the most
common modules for imaging. For a visual and for more detailed
information, see the `Optimization
Loop <https://mpol-dev.github.io/MPoL/tutorials/optimization.html>`__
tutorial or the MPoL SimpleNet `Source
Code <https://mpol-dev.github.io/MPoL/_modules/mpol/precomposed.html#SimpleNet>`__.
MPoL’s SimpleNet class is part of the ``mpol.precomposed`` library.

.. code:: ipython3

    from mpol.precomposed import SimpleNet
    model = SimpleNet(coords=coords, nchan=dataset.nchan)

We are now ready to move our model and data to the GPU. This process is
rather simple, PyTorch tensor objects are given a ``tensor.to(device)``
functionality that will move the data to the specific device. One can
also use the ``tensor.cuda()`` to move the tensor to the default CUDA
device. Both of these methods return a *copy* of the object on the GPU.
In our case, ``device='cuda:0'``, so we will move the *SimpleNet* object
(``model``) and our data (``dataset``), a GriddedDataset MPoL object, to
the GPU. Once we move ``model`` to the GPU, then we will create an
optimizer for the network.

Also defined below is the ``config`` dictionary. It is a set of
parameters used to scale our Neural Network’s learning. These specific
parameters are borrowed from the `Cross Validation
Tutorial <https://mpol-dev.github.io/MPoL/tutorials/crossvalidation.html>`__
due to the strong cross validation score they result in. For more
details on these variables, see the `Cross Validation
Tutorial <https://mpol-dev.github.io/MPoL/tutorials/crossvalidation.html>`__.

*Note: GriddedDataset objects also inherits a
``GriddedDataset.to(device)`` function that works similarly to
PyTorch’s*

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
        vis = model.forward()
        
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

Congratulations! You have now trained a neural network on your GPU. This
is the same SimpleNet as used in MPoL tutorial `Cross
Validation <https://mpol-dev.github.io/MPoL/tutorials/crossvalidation.html>`__.
As seen, the process for running on the GPU is rather simple. Once your
CUDA device has been set-up, it only requires moving the data and the
net to the GPU. 

