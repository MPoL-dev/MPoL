Installation
============

MPoL requires ``python >= 3.6``. 

Using pip
---------

Stable versions are hosted on PyPI and installable via ::

    $ pip install MPoL

From source
-----------

If you'd like to install the package from source to access the latest development version, download or ``git clone`` the MPoL repository and install ::

    $ git clone https://github.com/MPoL-dev/MPoL.git
    $ cd MPoL
    $ pip install .

If you have trouble installing please raise a `github issue <https://github.com/MPoL-dev/MPoL/issues>`_ with the particulars of your system.

If you're interested in contributing to the MPoL package, please see the :ref:`developer-documentation-label`.

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
