r"""
The gridding module contains routines to manipulate visibility data between uniform and non-uniform samples in the :math:`(u,v)` plane. You probably wont need to use these routines in the normal workflow of MPoL, but they are documented here for reference.
"""

import numpy as np
from scipy.sparse import lil_matrix
from mpol.datasets import UVDataset
from mpol.constants import *
import mpol.utils


def fftspace(width, N):
    """Delivers a (nearly) symmetric coordinate array that spans :math:`N` elements (where :math:`N` is even) from `-width` to `+width`, but ensures that the middle point lands on :math:`0`. The array indices go from :math:`0` to :math:`N -1.` 
    
    Args:
        width (float): the width of the array
        N (int): the number of elements in the array
        
    Returns:
        numpy.float64 1D array: the fftspace array
    
    """

    assert N % 2 == 0, "N must be even."

    dx = width * 2.0 / N
    xx = np.empty(N, np.float)
    for i in range(N):
        xx[i] = -width + i * dx

    return xx


def horner(x, a):
    """
    Use `Horner's method <https://introcs.cs.princeton.edu/python/21function/horner.py.html>`_ to compute and return the polynomial

    .. math::

        f(x) = a_0 + a_1 x^1 + a_2 x^2 + \ldots + a_{n-1} x^{(n-1)}

    Args:
        x (float): input
        a (list): list of polynomial coefficients 

    Returns:
        float: the polynomial evaluated at `x`
    """
    result = 0
    for i in range(len(a) - 1, -1, -1):
        result = a[i] + (x * result)
    return result


@np.vectorize
def spheroid(eta):
    r"""
    Prolate spheroidal wavefunction function assuming :math:`\alpha = 1` and :math:`m=6` for speed."

    Args:
        eta (float) : the value between ``[0, 1]``

    Returns:
        float : the value of the spheroid at :math:`\eta`
    """

    # Since the function is symmetric, overwrite eta
    eta = np.abs(eta)

    if eta <= 0.75:
        nn = eta ** 2 - 0.75 ** 2

        return horner(
            nn,
            np.array(
                [8.203343e-2, -3.644705e-1, 6.278660e-1, -5.335581e-1, 2.312756e-1]
            ),
        ) / horner(nn, np.array([1.0, 8.212018e-1, 2.078043e-1]))

    elif eta <= 1.0:
        nn = eta ** 2 - 1.0

        return horner(
            nn,
            np.array(
                [4.028559e-3, -3.697768e-2, 1.021332e-1, -1.201436e-1, 6.412774e-2]
            ),
        ) / horner(nn, np.array([1.0, 9.599102e-1, 2.918724e-1]))

    elif eta <= 1.0 + 1e-7:
        # case to allow some floating point error
        return 0.0

    else:
        # Now you're really outside of the bounds
        print(
            "The spheroid is only defined on the domain -1.0 <= eta <= 1.0. (modulo machine precision.)"
        )
        raise ValueError


def corrfun(eta):
    """
    The gridding *correction* function is applied to the image plane to counter-act the effects of the convolutional interpolation in the Fourier plane. For more information see `Schwab 1984 <https://ui.adsabs.harvard.edu/abs/1984iimp.conf..333S/abstract>`_. 

    Args:
        eta (float): the value in [0, 1]. Accepts floats or vectors of float.

    Returns:
        float: the correction function evaluated at :math:`\eta`
    """
    return spheroid(eta)


def corrfun_mat(alphas, deltas):
    """
    Calculate the image correction function over deminsionalities corresponding to the image. Apply to the image using a a multiply operation. The input coordinates must be fftshifted the same way as the image.

    Args:
        alphas (1D array): RA list 
        deltas (1D array): DEC list 

    Returns:
        2D array: correction function matrix evaluated over alphas and deltas

    """

    ny = len(deltas)
    nx = len(alphas)

    mat = np.empty((ny, nx), dtype=np.float64)

    # The size of one half-of the image.
    # sometimes ra and dec will be symmetric about 0, othertimes they won't
    # so this is a more robust way to determine image half-size
    maxra = np.abs(alphas[2] - alphas[1]) * nx / 2
    maxdec = np.abs(deltas[2] - deltas[1]) * ny / 2

    for i in range(nx):
        for j in range(ny):
            etax = (alphas[i]) / maxra
            etay = (deltas[j]) / maxdec

            if (np.abs(etax) > 1.0) or (np.abs(etay) > 1.0):
                # We would be querying outside the shifted image
                # bounds, so set this emission to 0.0
                mat[j, i] = 0.0
            else:
                mat[j, i] = 1 / (corrfun(etax) * corrfun(etay))

    return mat


def gcffun(eta):
    """
    The gridding *convolution* function for the convolution and interpolation of the visibilities in
    the Fourier domain. This is also the Fourier transform of ``corrfun``. For more information see `Schwab 1984 <https://ui.adsabs.harvard.edu/abs/1984iimp.conf..333S/abstract>`_.

    Args:
        eta (float): in the domain of [0,1]

    Returns:
        float : the gridding convolution function evaluated at :math:`\eta`
    """

    return np.abs(1.0 - eta ** 2) * spheroid(eta)


def calc_matrices(u_data, v_data, u_model, v_model):
    r"""
    Calculate real :math:`C_\Re` and imaginary :math:`C_\Im` sparse interpolation matrices for RFFT output, using spheroidal wave functions.

    Let :math:`W_\Re` and :math:`W_\Im` represent the real and imaginary output from the RFFT operatation carried out on an image that was pre-multiplied by the gridding correction function. To interpolate from the RFFT grid to the locations of `u_data` and `v_data`, one uses these matrices with a sparse matrix multiply to carry out 

    .. math::

        V_\Re = C_\Re W_\Re

    and 

    .. math::

        V_\Im = C_\Im W_\Im

    such that :math:`V_\Re` and :math:`V_\Im` are the real and imaginary visibilities interpolated to `u_data` and `v_data`. 

    Args:
    
        u_data (1D numpy array): the :math:`u` coordinates of the dataset (in [:math:`k\lambda`])
        v_data (1D numpy array): the :math:`v` coordinates of the dataset (in [:math:`k\lambda`])
        u_model: the :math:`u` axis delivered by the rfft (unflattened, in [:math:`k\lambda`]). Assuming this is trailing dimension, which is the one over which the RFFT was carried out.
        v_model: the :math:`v` axis delivered by the rfft (unflattened, in [:math:`k\lambda`]). Assuming this is leading dimension, which is the one over which the FFT was carried out.

    Returns:
        2-tuple : `coo` format sparse matrices :math:`C_\Re` and :math:`C_\Im`

    Begin with an image packed like ``Img[j, i]``, where ``i`` is the :math:`l` index and ``j`` is the :math:`m` index.
    Then the RFFT output will look like ``RFFT[j, i]``, where ``i`` is the u index and ``j`` is the v index.

    This assumes the `u_model` array is like the output from `np.fft.rfftfreq <https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfftfreq.html#numpy.fft.rfftfreq>`_ ::

        f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even

    and that the `v_model` array is like the output from `np.fft.fftfreq <https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftfreq.html>`_ ::

        f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even

    """

    # calculate the stride needed to advance one v point in the flattened array = the length of the u row
    vstride = len(u_model)
    Npix = len(v_model)


    data_points = np.array([u_data, v_data]).T
    # assert that the maximum baseline is contained within the model grid, which is the same as saying 
    # that the image-plane pixels are small enough.
    assert np.max(np.abs(u_data)) < u_model[-4] # choose -4, because we'll need 3 pixels to do the interpolation
    assert np.max(np.abs(v_data)) < v_model[Npix//2 - 4] # choose -4, because we'll need 3 pixels to do the interpolation

    # number of visibility points in the dataset
    N_vis = len(data_points)


    # initialize two sparse lil matrices for the instantiation
    # convert to csc at the end
    C_real = lil_matrix((N_vis, (Npix * vstride)), dtype=np.float64)
    C_imag = lil_matrix((N_vis, (Npix * vstride)), dtype=np.float64)

    # determine model grid spacing
    du = np.abs(u_model[1] - u_model[0])
    dv = np.abs(v_model[1] - v_model[0])

    # for each data_point within the grid, calculate the row and insert it into the matrix
    for row_index, (u, v) in enumerate(data_points):

        # assuming the grid stretched for -/+ values easily
        i0 = np.int(np.ceil(u / du))
        j0 = np.int(np.ceil(v / dv))

        i_indices = np.arange(i0 - 3, i0 + 3)
        j_indices = np.arange(j0 - 3, j0 + 3)

        # calculate the etas and weights
        u_etas = (u / du - i_indices) / 3
        v_etas = (v / dv - j_indices) / 3

        uw = gcffun(u_etas)
        vw = gcffun(v_etas)

        w = np.sum(uw) * np.sum(vw)

        l_indices = np.zeros(36, dtype=np.int)
        weights_real = np.zeros(36, dtype=np.float)
        weights_imag = np.zeros(36, dtype=np.float)

        # loop through all 36 points and calculate the matrix element
        # do it in this order because u indices change the quickest
        for j in range(6):
            for i in range(6):
                k = j * 6 + i

                i_index = i_indices[i]

                if i_index >= 0:
                    j_index = j_indices[j]
                    imag_prefactor = 1.0

                else:  # i_index < 0:
                    # map negative i index to positive i_index
                    i_index = -i_index

                    # map j index to opposite sign
                    j_index = -j_indices[j]

                    # take the complex conjugate for the imaginary weight
                    imag_prefactor = -1.0

                # shrink j to fit in the range of [0, Npix]
                if j_index < 0:
                    j_index += Npix

                l_indices[k] = i_index + j_index * vstride
                weights_real[k] = uw[i] * vw[j] / w
                weights_imag[k] = imag_prefactor * uw[i] * vw[j] / w

        # TODO: at the end, decide whether there is overlap and consolidate
        l_sorted, unique_indices, unique_inverse, unique_counts = np.unique(
            l_indices, return_index=True, return_inverse=True, return_counts=True
        )
        if len(unique_indices) < 36:
            # Some indices are querying the same point in the RFFT grid, and their weights need to be
            # consolidated

            N_unique = len(l_sorted)
            condensed_weights_real = np.zeros(N_unique)
            condensed_weights_imag = np.zeros(N_unique)

            # find where unique_counts > 1
            ind_multiple = unique_counts > 1

            # take the single weights from the array
            ind_single = ~ind_multiple

            # these are the indices back to the original weights of sorted singles
            # unique_indices[ind_single]

            # get the weights of the values that only occur once
            condensed_weights_real[ind_single] = weights_real[
                unique_indices[ind_single]
            ]
            condensed_weights_imag[ind_single] = weights_imag[
                unique_indices[ind_single]
            ]

            # the indices that occur multiple times
            # l_sorted[ind_multiple]

            # the indices of indices that occur multiple times
            ind_arg = np.where(ind_multiple)[0]

            for repeated_index in ind_arg:
                # figure out the indices of the repeated indices in the original flattened index array
                repeats = np.where(l_sorted[repeated_index] == l_indices)

                # stuff the sum of these weights into the condensed array
                condensed_weights_real[repeated_index] = np.sum(weights_real[repeats])
                condensed_weights_imag[repeated_index] = np.sum(weights_imag[repeats])

            # remap these variables to the shortened arrays
            l_indices = l_sorted
            weights_real = condensed_weights_real
            weights_imag = condensed_weights_imag

        C_real[row_index, l_indices] = weights_real
        C_imag[row_index, l_indices] = weights_imag

    return C_real.tocoo(), C_imag.tocoo()


def grid_datachannel(uu, vv, weights, re, im, cell_size, npix, debug=False, **kwargs):
    """
    Rather than interpolating the complex model visibilities from these grid points to the non-uniform :math:`(u,v)` points, pre-average the data visibilities to the nearest grid point. This saves time by eliminating an interpolation operation after every new model evaluation, since the model visibilities correspond to the locations of the gridded visibilities.

    Args:
        uu (list): the uu points (in [:math:`k\lambda`])
        vv (list): the vv points (in [:math:`k\lambda`])
        weights (list): the thermal weights (in [:math:`\mathrm{Jy}^{-2}`])
        re (list): the real component of the visibilities (in [:math:`\mathrm{Jy}`])
        im (list): the imaginary component of the visibilities (in [:math:`\mathrm{Jy}`])
        cell_size (float): the image cell size (in arcsec)
        npix (int): the number of pixels in each dimension of the square image
        debug (bool): provide the full 2D grid output of the gridded reals, imaginaries, and weights. Default False. 

    Returns:
        6-tuple: (`u_grid`, `v_grid`, `grid_mask`, `avg_weights`, `avg_re`, `avg_im`) tuple of arrays. `grid_mask` has shape (npix, npix//2 + 1) corresponding to the RFFT output of an image with `cell_size` and dimensions (npix, npix). The remaining arrays are 1D and have length corresponding to the number of true elements in `ind`. They correspond to the model visibilities when the RFFT output is indexed with `grid_mask`. If debug is true, return a 7-tuple which includes the number of visibilities per cell.

    An image `cell_size` and `npix` correspond to particular `u_grid` and `v_grid` values from the RFFT. 

    This pre-gridding procedure is similar to "uniform" weighting of visibilities for imaging, but not exactly the same in practice. This is because we are still evaluating a visibility likelihood function which incorporates the uncertainties of the measured spatial frequencies, whereas imaging routines use the weights to adjust the contributions of the measured visibility to the synthesize image. Evaluating a model against these gridded visibilities should be equivalent to the full interpolated calculation so long as it is true that 
    
        1) the visibility function is approximately constant over a :math:`(u,v)` cell 
        2) the measurement uncertainties on the real and imaginary components of individual visibilities are correct and described by Gaussian noise

    If (1) is violated, you can always increase the width and npix of the image (keeping `cell_size` constant) to shrink the size of the :math:`(u,v)` cells (i.e., see https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfftfreq.html). If you need to do this, this probably indicates that you weren't using an image size appropriate for your dataset. 
    
    If (2) is violated, then it's not a good idea to procede with this faster routine. Instead, revisit the calibration of your dataset, or build in self-calibration adjustments based upon antenna and time metadata of the visibilities. One downside of the pre-gridding operation is that it eliminates the possibility of using self-calibration type loss functions.
    """

    assert npix % 2 == 0, "Image must have an even number of pixels"

    assert np.max(np.abs(uu)) < mpol.utils.get_max_spatial_freq(cell_size, npix), "Dataset contains uu spatial frequency measurements larger than those in the proposed model image."
    assert np.max(np.abs(vv)) < mpol.utils.get_max_spatial_freq(cell_size, npix), "Dataset contains vv spatial frequency measurements larger than those in the proposed model image."

    assert np.all(weights > 0), "some weights are negative, check input args"


    # calculate the grid spacings
    cell_size = cell_size * arcsec  # [radians]
    # cell_size is also the differential change in sky angles
    # dll = dmm = cell_size #[radians]

    # the output spatial frequencies of the RFFT routine
    uu_grid = np.fft.rfftfreq(npix, d=cell_size) * 1e-3  # convert to [kλ]
    vv_grid = np.fft.fftfreq(npix, d=cell_size) * 1e-3  # convert to [kλ]

    nu = len(uu_grid)
    nv = len(vv_grid)

    du = np.abs(uu_grid[1] - uu_grid[0])
    dv = np.abs(vv_grid[1] - vv_grid[0])

    # The RFFT outputs u in the range [0, +] and v in the range [-, +],
    # but the dataset contains measurements at u [-,+] and v [-, +].
    # Find all the u < 0 points and convert them via complex conj
    ind_u_neg = uu < 0.0
    uu[ind_u_neg] *= -1.0  # swap axes so all u > 0
    vv[ind_u_neg] *= -1.0  # swap axes
    im[ind_u_neg] *= -1.0  # complex conjugate

    # calculate the sum of the weights within each cell
    # create the cells as edges around the existing points
    # note that at this stage, the bins are strictly increasing
    # when in fact, later on, we'll need to put this into fftshift format for the RFFT
    weight_cell, v_edges, u_edges = np.histogram2d(
        vv,
        uu,
        bins=[nv, nu],
        range=[
            (np.min(vv_grid) - dv / 2, np.max(vv_grid) + dv / 2),
            (np.min(uu_grid) - du / 2, np.max(uu_grid) + du / 2),
        ],
        weights=weights,
    )

    if debug:
        nvis, v_edges, u_edges = np.histogram2d(
        vv,
        uu,
        bins=[nv, nu],
        range=[
            (np.min(vv_grid) - dv / 2, np.max(vv_grid) + dv / 2),
            (np.min(uu_grid) - du / 2, np.max(uu_grid) + du / 2),
        ]
    )
    

    # calculate the weighted average and weighted variance for each cell
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean
    # also Bevington, "Data Reduction in the Physical Sciences", pg 57, 3rd ed.
    # where weight = 1/sigma^2

    # blank out the cells that have zero counts
    weight_cell[(weight_cell == 0.0)] = np.nan
    ind_ok = ~np.isnan(weight_cell)

    # weighted_mean = np.sum(x_i * weight_i) / weight_cell
    real_part, v_edges, u_edges = np.histogram2d(
        vv,
        uu,
        bins=[nv, nu],
        range=[
            (np.min(vv_grid) - dv / 2, np.max(vv_grid) + dv / 2),
            (np.min(uu_grid) - du / 2, np.max(uu_grid) + du / 2),
        ],
        weights=re * weights,
    )

    imag_part, v_edges, u_edges = np.histogram2d(
        vv,
        uu,
        bins=[nv, nu],
        range=[
            (np.min(vv_grid) - dv / 2, np.max(vv_grid) + dv / 2),
            (np.min(uu_grid) - du / 2, np.max(uu_grid) + du / 2),
        ],
        weights=im * weights,
    )

    weighted_mean_real = real_part / weight_cell
    weighted_mean_imag = imag_part / weight_cell

    # do an fftshift on weighted_means and sigmas to get this into a 2D grid
    # that matches the RFFT output directly

    ind = np.fft.fftshift(ind_ok, axes=0)  # RFFT indices that are not nan
    # return ind as a 2D grid, so we can index the model RFFT output to directly
    # compare to these values

    # list of gridded visibilities
    if debug:
        avg_weights = np.fft.fftshift(weight_cell, axes=0)
        avg_re = np.fft.fftshift(weighted_mean_real, axes=0)
        avg_im = np.fft.fftshift(weighted_mean_imag, axes=0)
        nvis = np.fft.fftshift(nvis, axes=0)

        return (uu_grid, vv_grid, ind, avg_weights, avg_re, avg_im, nvis)

    else:
        avg_weights = np.fft.fftshift(weight_cell, axes=0)[ind]
        avg_re = np.fft.fftshift(weighted_mean_real, axes=0)[ind]
        avg_im = np.fft.fftshift(weighted_mean_imag, axes=0)[ind]

        return (uu_grid, vv_grid, ind, avg_weights, avg_re, avg_im)



def grid_dataset(uus, vvs, weights, res, ims, cell_size, npix, **kwargs):
    """
    Pre-grid a dataset containing multiple channels to the expected `u_grid` and `v_grid` points from the RFFT routine. 

    Note that `nvis` need not be the same for each channel (i.e., `uus`, `vvs`, etc... can be ragged arrays, as long as each is iterable across the channel dimension). This routine iterates through the channels, calling :meth:`mpol.gridding.grid_datachannel` for each one.

    Args:
        uus (nchan, nvis) list: the uu points (in klambda)
        vvs (nchan, nvis) list: the vv points (in klambda)
        weights (nchan, nvis) list: the thermal weights
        res (nchan, nvis) list: the real component of the visibilities
        ims (nchan, nvis) list: the imaginary component of the visibilities
        cell_size: the image cell size (in arcsec)
        npix: the number of pixels in each dimension of the square image

    Returns:
        6-tuple: (`u_grid`, `v_grid`, `grid_mask`, `avg_weights`, `avg_re`, `avg_im`) tuple of arrays. `u_grid` and `v_grid` are 1D arrays representing the coordinates of the RFFT grid, and are provided for reference. `grid_mask` has shape (npix, npix//2 + 1) corresponding to the RFFT output of an image with `cell_size` and dimensions (npix, npix). The remaining arrays are 1D and have length corresponding to the number of true elements in `ind`. They correspond to the model visibilities when the RFFT output is indexed with `grid_mask`.

    """

    nchan = uus.shape[0]

    # pre-allocate the index array
    ind = np.zeros((nchan, npix, npix // 2 + 1), dtype="bool")

    avg_weights = []
    avg_re = []
    avg_im = []

    for i in range(nchan):
        uu_temp, vv_temp, ind_temp, w_temp, re_temp, im_temp = grid_datachannel(
            uus[i], vvs[i], weights[i], res[i], ims[i], cell_size, npix
        )
        ind[i] = ind_temp
        avg_weights.append(w_temp)
        avg_re.append(re_temp)
        avg_im.append(im_temp)

    # flatten all visibilities to a single vector
    # these are done because, once gridded, it's not certain that there will
    # be the same number of nvis per channel, depending on where the visibilities
    # fall in u,v space and to which cell they are gridded.
    avg_weights = np.concatenate(avg_weights)
    avg_re = np.concatenate(avg_re)
    avg_im = np.concatenate(avg_im)

    # just take the last uu and vv values, since they are the same for all channels

    return (uu_temp, vv_temp, ind, avg_weights, avg_re, avg_im)
