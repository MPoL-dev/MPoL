from __future__ import annotations

from typing import Final

import astropy.constants as apy_const
import numpy as np

# convert from arcseconds to radians
arcsec: Final[float] = np.pi / (180.0 * 3600)  # [radians]  = 1/206265 radian/arcsec

deg: Final[float] = np.pi / 180  # [radians]

# Boltzmann constant
kB: Final[float] = getattr(apy_const, "k_B").cgs.value  # [erg K^-1]

# Light speed
cc: Final[float] = getattr(apy_const, "c").cgs.value  # [cm s^-1]
c_ms: Final[float] = getattr(apy_const, "c").value  # [m s^-1]
