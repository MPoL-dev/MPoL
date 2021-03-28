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
* The "rows" of the image cube (axis=1) correspond to the :math:`m` or Dec axis. There are :math:`M` number of pixels in the Dec axis.
* The "columns" of the image cube (axis=2) correspond to the :math:`l` or R.A. axis. There are :math:`L` number of pixels in the R.A. axis.


-------------
Angular units
-------------

For nearly all user-facing routines, the angular axes (the direction cosines :math:`l` and :math:`m` corresponding to R.A. and Dec, respectively) are expected to have units of arcseconds. The sample rate of the image cube is set via the ``cell_size`` parameter, in units of arcsec. Internally, these quantities are represented in radians.  Let the pixel spacing be represented by :math:`\Delta l` and :math:`\Delta m`, respectively.

-------------
Fourier units
-------------

The sampling rate in the Fourier domain is inversely related to the number of samples and the sampling rate in the image domain. I.e., the grid spacing is 

.. math::

    \Delta u = \frac{1}{L \Delta l} \\
    \Delta v = \frac{1}{M \Delta m}. 

If :math:`\Delta l` and :math:`\Delta m` are in units of radians, then :math:`\Delta u` and :math:`\Delta v` are in units of cycles per radian. Thanks to the geometric relationship of the interferometer, the spatial frequency units can equivalently be expressed as the baseline lengths measured in multiples of the observing wavelength :math:`\lambda`. 

For example, take an observation with ALMA band 6 at an observing frequency of 230 GHz, corresponding to a wavelength of 1.3mm. A 100 meter baseline between antennas will measure a spatial frequency of :math:`\frac{100\,\mathrm{m} }{ 1.3 \times 10^{-3}\,\mathrm{m}} \approx 77 \mathrm{k}\lambda` or 77,000 cycles per radian.

For more information on the relationship between baselines and spatial frequencies, see `TMS Eqn Chapter 2.3, equations 2.13 and 2.14 <https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract>`_. Internally, MPoL usually represents spatial frequencies in units of :math:`\mathrm{k}\lambda`.

For reference, here are some typical ALMA baseline lengths and their (approximate) corresponding spatial frequencies at common observing frequencies

.. csv-table::
    :file: _static/baselines/build/baselines.csv
    :header-rows: 1

Occasionally, it is useful to represent the cartesian Fourier coordinates :math:`u`, :math:`v` in polar coordinates :math:`q`, :math:`\phi`

.. math::

    q = \sqrt{u^2 + v^2}\\
    \phi = \mathrm{atan2}(v,u).

------------------------------
The discrete Fourier transform
------------------------------

Since we are dealing with discrete quantities (pixels), we use the discrete versions of the Fourier transform (DFT), carried out by the Fast Fourier transform (FFT). Throughout the package we use the implementations in numpy or PyTorch: they are mathematically the same, but PyTorch provides the opportunities for autodifferentiation. For both the forward and inverse transforms, we assume that ``norm='backward'``, the default setting. This means we don't need to keep account for the :math:`L` or :math:`M` prefactors for the forward transform, but we do need to account for the :math:`U` and :math:`V` prefactors in the inverse transform.

**Forward transform**: As before, we use the forward transform to go from the image plane (sky brightness distribution) to the Fourier plane (visibility function). This is the most common transform used in MPoL because RML can be thought of as a type of forward modeling procedure: we're proposing an image and carrying it to the visibility plane to check its fit with the data. In numpy, the forward FFT is `defined as <https://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft>`_ 

.. math::

    \mathtt{FFT}(I_{l,m}) = \sum_{l=0}^{L-1} \sum_{m=0}^{M-1} I_{l,m} \exp \left \{- 2 \pi i (ul/L + vm/M) \right \}

To make the FFT output an appropriate representation of the continuous forward Fourier transform, we need to account for the spacing of the input samples. The FFT knows only that it was served a sequence of numbers, it does not know that the samples in :math:`I_{l,m}` are spaced ``cell_size`` apart. To do this, we just need to account for the spacing as a prefactor (i.e., converting the :math:`\mathrm{d}l` to :math:`\Delta l`), following `TMS Eqn A8.18 <https://ui.adsabs.harvard.edu/abs/2017isra.book.....T/abstract>`_

.. math::
    
    V_{u,v} = (\Delta l)(\Delta m) \mathtt{FFT}(I_{l,m})

In this context, the :math:`u,v` subscripts indicate the elements of the :math:`V` array. As long as :math:`I_{l,m}` is in units of :math:`\mathrm{Jy} / (\Delta l \Delta m)`, then :math:`V` will be in the correct output units (flux, or Jy).

**Inverse transform**: The inverse transform is used within MPoL to produce a quick diagnostic image from the visibilities (called the "dirty image"). As you might expect, this is the inverse operation of the forward transform. Numpy and PyTorch define the inverse transform as

.. math::

    \mathtt{iFFT}({\cal V}_{u,v}) = \frac{1}{U} \frac{1}{V} \sum_{l=0}^{U-1} \sum_{m=0}^{V-1} {\cal V}_{u,v} \exp \left \{2 \pi i (ul/L + vm/M) \right \}

If we had a fully sampled grid of :math:`{\cal V}_{u,v}` values, then the operation we'd want to carry out to produce an image needs to correct for both the cell spacing and the counting terms

.. math::

    I_{l,m} = U V (\Delta u)(\Delta v) \mathtt{iFFT}({\cal V}_{u,v})

For more information on this procedure as implmented in MPoL, see the :class:`~mpol.gridding.Gridder` class and the source code of its :func:`~mpol.gridding.Gridder.get_dirty_image` method. When the grid of :math:`{\cal V}_{u,v}` values is not fully sampled (as in any real-world interferometric observation), there are many subtleties beyond this simple equation that warrant consideration when synthesizing an image via inverse Fourier transform. For more information, consult the seminal `Ph.D. thesis <http://www.aoc.nrao.edu/dissertations/dbriggs/>`_ of Daniel Briggs.