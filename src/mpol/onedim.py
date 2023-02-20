import numpy as np 
import matplotlib.pyplot as plt 

from mpol.fourier import NuFFT
from mpol.utils import torch2npy


def get_1d_vis_fit(model, u, v, chan=0):
    r"""
    Obtain the 1D (radial) visibility model V(q) corresponding to a 2D MPoL 
    image-domain model. 

    Parameters
    ----------
    model : `torch.nn.Module` object
        Instance of the `mpol.precomposed.SimpleNet` class
    u : array, unit=:math:[`k\lambda`] 
        u-coordinates at which to sample (e.g., those of the dataset)
    v : array, unit=:math:[`k\lambda`]
        v-coordinates at which to sample (e.g., those of the dataset)
    chan : int, default=0
        Channel of `model` to select

    Returns
    -------
    q : array, unit=:math:[`k\lambda`]
        Baselines corresponding to `u` and `v`
    Vmod : array, unit=[Jy] # TODO: right unit?
        Visibility amplitudes at `q`
    """
    q = np.hypot(u, v)

    nufft = NuFFT(coords=model.coords, nchan=model.nchan, uu=u, vv=v)
    # get model visibilities 
    Vmod = nufft(model.icube()).detach()[chan] 
    
    return q, Vmod

