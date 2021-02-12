Units and Conventions
=====================

Fourier transform conventions
-----------------------------

We follow the (reasonably standard) conventions of the Fourier transform (e.g., `Bracewell's <https://ui.adsabs.harvard.edu/abs/2000fta..book.....B/abstract>`_ "system 1"). 

**Forward transform**:

.. math::

    F(s) = \int_{-\infty}^\infty f(x) e^{-i 2 \pi  x s} \mathrm{d}x

**Inverse transform**:

.. math::

    f(x) = \int_{-\infty}^\infty F(s) e^{i 2 \pi  x s} \mathrm{d}s


Continuous representation of interferometry
-------------------------------------------

Consider some astronomical source parameterized by its sky brightness distribution :math:`I`. The sky brightness is a function of position on the sky. For small fields of view (typical to single-pointing ALMA or VLA observations) we parameterize the sky direction using the direction cosines :math:`l` and :math:`m`, which correspond to R.A. and Dec, respectively. In that case, we would have a function :math:`I(l,m)`. The sky brightness is an *intensity*, so it has units of :math:`\mathrm{Jy\,arcsec}^{-2}` (equivalently :math:`\mathrm{Jy\, beam}^{-1}`, where :math:`\mathrm{beam}` is the effective area of the synthesized beam).

The real domain is linked to the Fourier domain, also called the visibility domain, via the Fourier transform 

.. math::

    {\cal V}(u,v) = \int \int I(l,m) \exp \left \{- 2 \pi i (ul + vm) \right \} \, \mathrm{d}l\,\mathrm{d}m.

This integral demonstrates that the units of visibility function (and samples of it) are :math:`\mathrm{Jy}`.

Discretized representation 
--------------------------

There are several annoying pitfalls that can arise when dealing with discretized images and Fourier transforms, and most relate back to confusing or ill-specified conventions. The purpose of this page is to explicitly define the conventions used throughout MPoL and make clear how each transformation relates back to the continuous equations.

------------
Pixel fluxes 
------------

* Throughout the codebase, any sky plane cube representing the sky plane is assumed to have units of :math:`\mathrm{Jy\,arcsec}^{-2}`. 
* The image cubes are packed as 3D arrays ``(nchan, npix, npix)``. 
* The "rows" of the image cube (axis=1) correspond to the :math:`m` or Dec axis 
* The "columns" of the image cube (axis=2) correspond to the :math:`l` or R.A. axis


-------------
Angular units
-------------

For nearly all user-facing routines, the angular axes (the direction cosines :math:`l` and :math:`m` corresponding to R.A. and Dec, respectively) are expected to have units of arcseconds. Internally, these quantities are represented in radians. The sample rate of the image cube is set via the ``cell_size`` parameter, in units of arcsec.

------------------------------
The discrete Fourier transform
------------------------------

Since we are dealing with discrete quantities (pixels), we use the discrete Fourier transform (DFT), carried out by the Fast Fourier transform (FFT). In numpy, the forward FFT is `defined as <https://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft>`_ 

.. math::

    \mathtt{FFT}(I_{l,m}) = \sum_{l=0}^{L-1} \sum_{m=0}^{M-1} I_{l,m} \exp \left \{- 2 \pi i (ul/L + vm/M) \right \}

To make the FFT output an appropriate representation of the continuous Fourier transform, we need to account for the spacing of the input samples. The FFT knows only that it was served a sequence of numbers, it does not know that the samples in :math:`I_{l,m}` are spaced ``cell_size`` apart. To do this, we just need to account for the spacing as a prefactor (i.e., converting the :math:`\mathrm{d}l` to :math:`\Delta l`), following `TMS Eqn A8.18 <https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract>`_

.. math::
    
    V_{u,v} = (\Delta l)(\Delta m) \mathtt{FFT}(I_{l,m})

In this context, the :math:`u,v` subscripts indicate the elements of the :math:`V` array. As long as :math:`I_{l,m}` is in units of :math:`\mathrm{Jy} / (\Delta l \Delta m)`, then :math:`V` will be in the correct output units (flux, or Jy).
