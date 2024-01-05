import numpy as np
from astropy.constants import c, k_B

# convert from arcseconds to radians
arcsec: float = np.pi / (180.0 * 3600)  # [radians]  = 1/206265 radian/arcsec

deg: float = np.pi / 180  # [radians]

kB: float = k_B.cgs.value  # [erg K^-1] Boltzmann constant
cc: float = c.cgs.value  # [cm s^-1]
c_ms: float = c.value  # [m s^-1]
