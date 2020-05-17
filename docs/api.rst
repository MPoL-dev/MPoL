===
API
===

This page documents all of the available components of the MPoL package. If you do not see something that you think should be documented, please raise an `issue <https://github.com/iancze/MPoL/issues>`_.

A note about units
------------------

Real domain 
===========

------------
Pixel fluxes 
------------

Internally, the image cube is represented as the natural logarithm of the pixel intensities (:attr:`ImageCube._log_cube`)

.. math::

    s_{l,m,v} = \ln(I_{l,m,v})

such that 

.. math::

    I_{l,m,v} = \exp(s_{l,m,v})

where the :math:`l,m,v` subscripts represent the direction cosines and velocity dimension, respectively. The convenience method :attr:`ImageCube.cube` is provided to do this conversion naturally.

Throughout the codebase, :attr:`ImageCube.cube` is assumed to have units of :math:`\mathrm{Jy\,arcsec}^{-2}`. Note that this is in contrast to the units commonly used in radio astronomy, :math:`\mathrm{Jy\, beam}^{-1}`, where :math:`\mathrm{beam}` is the effective area of the synthesized beam. Regardless of which unit it is quoted in, :math:`I_{l,m,v}` is technically an intensity (not a flux).

-------------
Angular units
-------------

For nearly all user-facing routines, the angular axes (the direction cosines :math:`l` and :math:`m` corresponding to R.A. and Dec, respectively) are expected to have units of arcseconds. Internally, these quantities are represented in radians. The sample rate of the image cube is set via the ``cell_size`` parameter, in units of arcsec.


Fourier domain
==============

The real domain is linked to the Fourier domain, also called the visibility domain, via the Fourier transform 

.. math::

    {\cal V}(u,v) = \int \int I(l,m) \exp \left \{- 2 \pi i (ul + vm) \right \} \, \mathrm{d}l\,\mathrm{d}m.

Since we are dealing with discrete quantities (pixels), we use the discrete Fourier transform (DFT), carried out by the Fast Fourier transform. In numpy, the DFT is `defined as <https://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft>`_ 

.. math::

    \mathtt{DFT}(I_{l,m}) = \sum_{l=0}^{L-1} \sum_{m=0}^{M-1} I_{l,m} \exp \left \{- 2 \pi i (ul/L + vm/M) \right \}

To make the DFT output an appropriate representation of the continuous Fourier transform, we need to account for the spacing of the input samples. The DFT knows only that it was served a sequence of numbers, it does not know that the samples in :math:`I_{l,m}` are spaced ``cell_size`` apart. To do this, we just need to account for the spacing as a prefactor (i.e., converting the :math:`\mathrm{d}l` to :math:`\Delta l`), following `TMS Eqn A8.18 <https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract>`_

.. math::
    
    V_{u,v} = (\Delta l)(\Delta m) \mathtt{DFT}(I_{l,m})

In this context, the :math:`u,v` subscripts indicate the elements of the :math:`V` array. As long as :math:`I_{l,m}` is in units of :math:`\mathrm{Jy} / (\Delta l \Delta m)`, then :math:`V` will be in the correct output units (flux, or Jy).

Images 
------

.. automodule:: mpol.images
    :members:

Losses
------

.. automodule:: mpol.losses
    :members:

Datasets 
--------

.. automodule:: mpol.datasets
    :members:


Utilities
---------

.. automodule:: mpol.utils 
    :members:


Gridding
--------

.. automodule:: mpol.gridding
    :members: