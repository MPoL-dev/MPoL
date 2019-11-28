import numpy as np
from scipy.sparse import lil_matrix

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
