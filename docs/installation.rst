Installation
============

This package requires ``python > 3.5``. Install the requirements (`numpy`, `scipy`, `torch`, `torchvision`). You may consider `installing PyTorch individually <https://pytorch.org/>`_ if you do/do not require CUDA support and want to fine tune your installation. If you would like to export your images to FITS files, you should also install the `astropy` package.


Using pip
---------

After the dependencies are installed, ::

    $ pip install MPoL

From source
-----------

If you'd like to install the package from source, download or `git clone` the MPoL repository and install ::

    $ git clone https://github.com/iancze/MPoL.git
    $ cd MPoL
    $ python setup.py install

If you have trouble installing please raise a `github issue <https://github.com/iancze/MPoL/issues>`_.

Upgrading
---------

To upgrade to the latest stable version of MPoL, do ::

    $ pip install --upgrade MPoL

If you want to follow along with the latest developments, we recommend installing from source and tracking the ``master`` branch. Then, to update the repository ::

    $ cd MPoL
    $ git pull 
    $ python setup.py install

You can determine your current installed version by ::

    $ python 
    >>> import mpol 
    >>> print(mpol.__version__)