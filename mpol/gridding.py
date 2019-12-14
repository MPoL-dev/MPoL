import numpy as np
from scipy.sparse import lil_matrix
from mpol.datasets import UVDataset
from mpol.constants import *

# implementation of the gridding convolution functions in sparse matrix form
# also image pre-multiply (corrfun)


def fftspace(width, N):
    """Oftentimes it is necessary to get a symmetric coordinate array that spans ``N``
     elements from `-width` to `+width`, but makes sure that the middle point lands
     on ``0``. The indices go from ``0`` to ``N -1.``
     `linspace` returns  the end points inclusive, wheras we want to leave out the
     right endpoint, because we are sampling the function in a cyclic manner."""

    assert N % 2 == 0, "N must be even."

    dx = width * 2.0 / N
    xx = np.empty(N, np.float)
    for i in range(N):
        xx[i] = -width + i * dx

    return xx


def horner(x, a):
    """
    Use Horner's method to compute and return the polynomial
        a[0] + a[1] x^1 + a[2] x^2 + ... + a[n-1] x^(n-1)
    evaluated at x.

    from https://introcs.cs.princeton.edu/python/21function/horner.py.html
    """
    result = 0
    for i in range(len(a) - 1, -1, -1):
        result = a[i] + (x * result)
    return result


@np.vectorize
def spheroid(eta):
    """
        `spheroid` function which assumes ``\\alpha`` = 1.0, ``m=6``,  built for speed."

        Args:
            eta (float) : the value between [0, 1]
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
    Gridding *correction* function, but able to be passed either floating point numbers or vectors of `Float64`."

    Args:
        eta (float): the value in [0, 1]
    """
    return spheroid(eta)


def corrfun_mat(alphas, deltas):
    """
    Calculate the pre-multiply correction function to the image.
    Return as a 2D array.

    Args:
        alphas (1D array): RA list (pre-fftshifted)
        deltas (1D array): DEC list (pre-fftshifted)

    Returns:
        (2D array): correction function matrix evaluated over alphas and deltas

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
    The gridding *convolution* function, used to do the convolution and interpolation of the visibilities in
    the Fourier domain. This is also the Fourier transform of `corrfun`.

    Args:
        eta (float): in the domain of [0,1]
    """

    return np.abs(1.0 - eta ** 2) * spheroid(eta)


def calc_matrices(u_data, v_data, u_model, v_model):
    """
    Calcuate the real and imaginary interpolation matrices in one pass.

    Args:
        data_points: the pairs of u,v points in the dataset (in klambda)
        u_model: the u axis delivered by the rfft (unflattened). Assuming this is the RFFT axis.
        v_model: the v axis delivered by the rfft (unflattened). Assuming this is the FFT axis.

    Start with an image packed like Img[j, i]. i is the alpha index and j is the delta index.
    Then the RFFT output will have RFFT[j, i]. i is the u index and j is the v index.

    see also `Model.for` routine in MIRIAD source code.
    Uses spheroidal wave functions to interpolate a model to a (u,v) coordinate.
    Ensure that datapoints and u_model,v_model are in consistent units (either λ or kλ).

    (m, ..., n//2+1, 2)
    u freqs stored like this
    f = [0, 1, ...,     n/2-1,     n/2] / (d*n)   if n is even

    v freqs stored like this
    f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] / (d*n)   if n is even

    """

    data_points = np.array([u_data, v_data]).T
    # TODO: assert that the maximum baseline is contained within the model grid.
    # TODO: assert that the image-plane pixels are small enough.

    # number of visibility points in the dataset
    N_vis = len(data_points)

    # calculate the stride needed to advance one v point in the flattened array = the length of the u row
    vstride = len(u_model)
    Npix = len(v_model)

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


def grid_datachannel(uu, vv, weights, re, im, cell_size, npix):
    """
    Pre-grid a single-frequency dataset to the expected `u_grid` and `v_grid` points from the RFFT routine to save on interpolation costs. 

    Args:
        uu (nvis) list: the uu points (in klambda)
        vv (nvis) list: the vv points (in klambda)
        weights (nvis) list: the thermal weights
        re (nvis) list: the real component of the visibilities
        im (nvis) list: the imaginary component of the visibilities
        cell_size: the image cell size (in arcsec)
        npix: the number of pixels in each dimension of the square image

    Returns:
        (ind, avg_weights, avg_re, avg_im) tuple of arrays. `ind` has shape (npix, npix//2 + 1). This shape corresponds to the RFFT output of an image with `cell_size` and dimensions (npix, npix). The remaining arrays are 1D and have length corresponding to the number of true elements in `ind`.

    An image `cell_size` and `npix` correspond to particular `u_grid` and `v_grid` values from the RFFT. Rather than interpolating the complex model visibilities from these grid points to the individual (u,v) points, pre-average the data visibilities to the nearest grid point. This means that there doesn't need to be an interpolation operation after every new model evaluation, since the model visibilities directly correspond to the locations of the gridded visibilities.

    This procedure is similar to "uniform" weighting of visibilities for imaging, but not exactly the same in practice. This is because we are still evaluating a visibility likelihood function which incorporates the uncertainties of the measured spatial frequencies (imaging routines do not use the uncertainties in quite the same way). Evaluating a model against these gridded visibilities should be equivalent to the full interpolated calculation so long as it is true that 
    
        1) the visibility function is approximately constant over the (u,v) cell it was averaged
        2) the measurement uncertainties on the real and imaginary components of individual visibilities are correct and described by Gaussian noise

    If (1) is violated, you can always increase the width and npix of the image (keeping cell_size constant) to shrink the size of the (u,v) cells (i.e., see https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.rfftfreq.html). This is also probably indicative that you aren't using an image size appropriate for your dataset. 
    
    If (2) is violated, then it's not a good idea to procede with this faster routine. Instead, revisit the calibration of your dataset, or build in self-calibration using tweaks to amplitude gain factors. That said, in general it will not be possible to use self-calibration type loss functions with pre-gridded visibilities
    """

    assert npix % 2 == 0, "Image must have an even number of pixels"

    # calculate the grid spacings

    cell_size = cell_size * arcsec  # [radians]
    # cell_size is also the differential change in sky angles
    # dll = dmm = cell_size #[radians]

    # the output spatial frequencies of the RFFT routine
    u_grid = np.fft.rfftfreq(npix, d=cell_size) * 1e-3  # convert to [kλ]
    v_grid = np.fft.fftfreq(npix, d=cell_size) * 1e-3  # convert to [kλ]

    nu = len(u_grid)
    nv = len(v_grid)

    du = np.abs(u_grid[1] - u_grid[0])
    dv = np.abs(v_grid[1] - v_grid[0])

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
            (np.min(v_grid) - dv / 2, np.max(v_grid) + dv / 2),
            (np.min(u_grid) - du / 2, np.max(u_grid) + du / 2),
        ],
        weights=weights,
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
            (np.min(v_grid) - dv / 2, np.max(v_grid) + dv / 2),
            (np.min(u_grid) - du / 2, np.max(u_grid) + du / 2),
        ],
        weights=re * weights,
    )

    imag_part, v_edges, u_edges = np.histogram2d(
        vv,
        uu,
        bins=[nv, nu],
        range=[
            (np.min(v_grid) - dv / 2, np.max(v_grid) + dv / 2),
            (np.min(u_grid) - du / 2, np.max(u_grid) + du / 2),
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
    avg_weights = np.fft.fftshift(weight_cell, axes=0)[ind]
    avg_re = np.fft.fftshift(weighted_mean_real, axes=0)[ind]
    avg_im = np.fft.fftshift(weighted_mean_imag, axes=0)[ind]

    return (ind, avg_weights, avg_re, avg_im)


def grid_dataset(uus, vvs, weights, res, ims, cell_size, npix):
    """
    Pre-grid a dataset containing multiple channels to the expected `u_grid` and `v_grid` points from the RFFT routine to save on interpolation costs. 

    Note that nvis need not be the same for each channel (i.e., uu, vv, etc... can be a ragged array, as long as it is iterable across the channel dimension). This routine iterates through the channels, calling grid_datachannel for each one.

    Args:
        uu (nchan, nvis) list: the uu points (in klambda)
        vv (nchan, nvis) list: the vv points (in klambda)
        weights (nchan, nvis) list: the thermal weights
        re (nchan, nvis) list: the real component of the visibilities
        im (nchan, nvis) list: the imaginary component of the visibilities
        cell_size: the image cell size (in arcsec)
        npix: the number of pixels in each dimension of the square image

    Returns:
        (ind, avg_weights, avg_re, avg_im) tuple of arrays. `ind` has shape (nchan, npix, npix//2 + 1). This shape corresponds to the RFFT output of an image cube with `nchan`, `cell_size`, and dimensions (npix, npix). The remaining arrays are 1D and have length corresponding to the number of true elements in `ind`.
        
    """

    nchan = uus.shape[0]

    # pre-allocate the index array
    ind = np.zeros((nchan, npix, npix // 2 + 1), dtype="bool")

    avg_weights = []
    avg_re = []
    avg_im = []

    for i in range(nchan):
        ind_temp, w_temp, re_temp, im_temp = grid_datachannel(
            uus[i], vvs[i], weights[i], res[i], ims[i], cell_size, npix
        )
        ind[i] = ind_temp
        avg_weights.append(w_temp)
        avg_re.append(re_temp)
        avg_im.append(im_temp)

    # flatten all visibilities to a single vector
    avg_weights = np.concatenate(avg_weights)
    avg_re = np.concatenate(avg_re)
    avg_im = np.concatenate(avg_im)

    return (ind, avg_weights, avg_re, avg_im)
