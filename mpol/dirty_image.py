import matplotlib.pylab as plt
import numpy as np
import numpy.linalg as linalg
from numpy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift, rfftfreq
from mpol.constants import *


def get_dirty_image(uu, vv, weights, re, im, cell_size, npix, robust=-2, **kwargs):
    r"""
    "Grid" the data visibilities using robust weighting, then do the inverse FFT. This delivers a maximum likelihood "dirty image" when robust=-2, corresponding to the uniform image. For decent image quality, however, robust values of 0.5 or 1.0 are recommended.

    Args:
        uu (list): the uu points (in [:math:`k\lambda`])
        vv (list): the vv points (in [:math:`k\lambda`])
        weights (list): the thermal weights (in [:math:`\mathrm{Jy}^{-2}`])
        re (list): the real component of the visibilities (in [:math:`\mathrm{Jy}`])
        im (list): the imaginary component of the visibilities (in [:math:`\mathrm{Jy}`])
        cell_size (float): the image cell size (in arcsec)
        npix (int): the number of pixels in each dimension of the square image
        robust (float): the Briggs robust parameter in the range [-2, 2]. -2 corresponds to uniform weighting while 2 corresponds to natural weighting.
        
    Returns:
        An image packed with the mm values increasing with row index and ll values decreasing with column index. If the image is plotted with `imshow` and `origin="lower"` it will display correctly.
        
    An image `cell_size` and `npix` correspond to particular `u_grid` and `v_grid` values from the RFFT. 

    """

    assert npix % 2 == 0, "Image must have an even number of pixels"
    assert (robust >= -2) and (robust <= 2), "Robust parameter must be in the range [-2, 2]"

    # calculate the grid spacings
    cell_size = cell_size * arcsec  # [radians]
    # cell_size is also the differential change in sky angles
    # dll = dmm = cell_size #[radians]

    # the output spatial frequencies of the FFT routine
    uu_grid = np.fft.rfftfreq(npix, d=cell_size) * 1e-3  # convert to [kλ]
    vv_grid = np.fft.fftfreq(npix, d=cell_size) * 1e-3  # convert to [kλ]

    nu = len(uu_grid)
    nv = len(vv_grid)

    du = np.abs(uu_grid[1] - uu_grid[0])
    dv = np.abs(vv_grid[1] - vv_grid[0])

    # expand and overwrite the vectors to include complex conjugates
    uu = np.concatenate([uu, -uu])
    vv = np.concatenate([vv, -vv])
    weights = np.concatenate([weights, weights])
    re = np.concatenate([re, re])
    im = np.concatenate([im, -im])  # the complex conjugates

    # The RFFT outputs u in the range [0, +] and v in the range [-, +],
    # but the dataset contains measurements at u [-,+] and v [-, +].
    # Find all the u < 0 points and convert them via complex conj
    ind_u_neg = uu < 0.0
    uu[ind_u_neg] *= -1.0  # swap axes so all u > 0
    vv[ind_u_neg] *= -1.0  # swap axes
    im[ind_u_neg] *= -1.0  # complex conjugate

    # implement robust weighting using the definition used in CASA
    # https://casa.nrao.edu/casadocs-devel/stable/imaging/synthesis-imaging/data-weighting

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

    # calculate the robust parameter f^2
    f_sq = ((5 * 10 ** (-robust)) ** 2) / (np.sum(weight_cell**2) / np.sum(weights))

    # the robust weight corresponding to the cell 
    cell_robust_weight = 1 / (1 + weight_cell * f_sq)

    # zero out cells that have no visibilities 
    cell_robust_weight[weight_cell <= 0.0] = 0

    # figure out which cell each visibility lands in, so that
    # we can assign it the appropriate robust weight for that cell
    # do this by calculating the nearest cell index [0, N] for all samples
    index_u = np.digitize(uu, u_edges)
    index_v = np.digitize(vv, v_edges)

    # now assign the cell robust weight to each visibility within that cell
    vis_robust_weight = cell_robust_weight[index_v, index_u]

    vis_total_weight = weights * vis_robust_weight

    real_part, v_edges, u_edges = np.histogram2d(
        vv,
        uu,
        bins=[nv, nu],
        range=[
            (np.min(vv_grid) - dv / 2, np.max(vv_grid) + dv / 2),
            (np.min(uu_grid) - du / 2, np.max(uu_grid) + du / 2),
        ],
        weights=re * vis_total_weight,
    )

    imag_part, v_edges, u_edges = np.histogram2d(
        vv,
        uu,
        bins=[nv, nu],
        range=[
            (np.min(vv_grid) - dv / 2, np.max(vv_grid) + dv / 2),
            (np.min(uu_grid) - du / 2, np.max(uu_grid) + du / 2),
        ],
        weights=im * vis_total_weight,
    )

    
    # gridded visibilities
    avg_re = np.fft.fftshift(real_part, axes=0)
    avg_im = np.fft.fftshift(imag_part, axes=0)

    # do the inverse FFT
    VV = avg_re + avg_im * 1.0j
    VV /= np.sum(vis_total_weight)

    im = np.fliplr(np.fft.fftshift(np.fft.irfftn(VV, axes=(0,1))))

    return im
