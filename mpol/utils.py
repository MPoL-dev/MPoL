import numpy as np

from mpol.constants import *


def get_Jy_ster(T_b, nu=230e9):
    """
    Get specific intensity from the brightness temperature.

    Args:
        T_b : brightness temperature in Kelvin
        nu : frequency (in Hz)

    Returns:
        specific intensity (in Jy/ster)
    """

    # brightness temperature assuming RJ limit
    # units of ergs/s/cm^2/Hz/ster
    I_nu = T_b * 2 * nu ** 2 * kB / cc ** 2

    # convert to Jy/ster
    Jy_ster = I_nu * 1e23

    return Jy_ster
