Installation
============

MPoL requires ``python > 3.5``. 

Using pip
---------

Stable versions are hosted on PyPI and installable via ::

    $ pip install MPoL

From source
-----------

If you'd like to install the package from source to access the latest development version, download or `git clone` the MPoL repository and install ::

    $ git clone https://github.com/iancze/MPoL.git
    $ cd MPoL
    $ pip install .

If you have trouble installing please raise a `github issue <https://github.com/iancze/MPoL/issues>`_.

Conda
-----

Automatic installation of depedencies in conda environments failed intermittently for some users. You might want to install the requirements first(`numpy`, `scipy`, `torch`, `torchvision`), and then `pip install .`

Upgrading
---------

If you installed from PyPI, to upgrade to the latest stable version of MPoL, do ::

    $ pip install --upgrade MPoL

If you installed from source, update the repository ::

    $ cd MPoL
    $ git pull 
    $ pip install .

You can determine your current installed version by ::

    $ python 
    >>> import mpol 
    >>> print(mpol.__version__)

Other considerations 
--------------------

You may consider `installing PyTorch individually <https://pytorch.org/>`_ if you do/do not require CUDA support and want to fine tune your installation. If you would like to export your images to FITS files, you should also install the `astropy` package. 