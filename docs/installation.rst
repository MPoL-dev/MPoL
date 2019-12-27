============
Installation
============

Requires ``python > 3.5``. Install the requirements (numpy, scipy, torch, torchvision). You may consider `installing PyTorch individually <https://pytorch.org/>`_ if you do/do not require CUDA support and want to fine tune your installation. Or, you could just do::

    $ pip install -r requirements.txt

If you would like to export your images to FITS files, you should also install the astropy package.

Then download or ``git clone`` the MPoL repository and install:::

    $ git clone https://github.com/iancze/MPoL.git
    $ cd MPoL
    $ python setup.py install

If you have trouble installing please raise a `github issue <https://github.com/iancze/MPoL/issues>`_.