import torch
import numpy as np

from .constants import arcsec, cc, kB


def get_Jy_arcsec2(T_b, nu=230e9):
    """
    Calculate specific intensity from the brightness temperature, using the Rayleigh-Jeans definition.

    Args:
        T_b : brightness temperature in [:math:`K`]
        nu : frequency (in Hz)

    Returns:
        float: specific intensity (in [:math:`\mathrm{Jy}\, \mathrm{arcsec}^2]`)
    """
    # brightness temperature assuming RJ limit
    # units of ergs/s/cm^2/Hz/ster
    I_nu = T_b * 2 * nu ** 2 * kB / cc ** 2

    # convert to Jy/ster
    Jy_ster = I_nu * 1e23

    # convert to Jy/arcsec^2
    Jy_arcsec2 = Jy_ster * arcsec ** 2

    return Jy_arcsec2


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


def fftshift(x, axes=None):
    """
    `fftshift <https://docs.scipy.org/doc/numpy/reference/generated/numpy.fft.fftshift.html>`_ the input array along each axis. For even-length arrays, fftshift and ifftshift are equivalent operations. 

    Args:
        x : a torch tensor 
        axes : tuple selecting which axes to shift over. Default is all.

    Returns:
        x : an fftshift-ed tensor
    """
    if axes is None:
        axes = range(0, len(x.size()))

    for dim in axes:
        x = torch.roll(x, x.size(dim) // 2, dims=dim)
    return x


def get_max_spatial_freq(cell_size, npix):
    """
    Calculate the maximum spatial frequency contained in the image.

    Args:
        cell_size (float): the pixel size in arcseconds
        npix (int): the number of pixels in the image

    Returns:
        max_freq : the maximum spatial frequency contained in the image (in kilolambda)
    """

    # technically this is as straightforward as doing 1/(2 * cell_size), but for even-sized
    # arrays, the highest *positive* spatial frequency is (npix/2 - 1) / (npix * cell_size)
    # it is the most negative spatial frequency that goes to - 1/(2 * cell_size)

    return (npix / 2 - 1) / (npix * cell_size * arcsec) * 1e-3  # kilolambda


def get_maximum_cell_size(uu_vv_point):
    """
    Calculate the maximum possible cell_size that will Nyquist sample the uu or vv point. Note: not q point.

    Args: 
        uu_vv_point (float): a single spatial frequency. Units of [:math:`\mathrm{k}\lambda`].

    Returns: cell_size (in arcsec)
    """

    return 1 / ((2 - 1) * uu_vv_point * 1e3) / arcsec

