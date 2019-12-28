import numpy as np
import torch

from mpol.constants import *


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
