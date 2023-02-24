import numpy as np 

from mpol.fourier import NuFFT
from mpol.geometry import observer_to_flat
from mpol.utils import torch2npy


def get_1d_vis_fit(model, chan=0):
    r"""
    Obtain the 1D (radial) visibility model V(q) corresponding to a 2D MPoL 
    image-domain model. 

    Parameters
    ----------
    model : `torch.nn.Module` object
        Instance of the `mpol.precomposed.SimpleNet` class
    chan : int, default=0
        Channel of `model` to select

    Returns
    -------
    q : array, unit=:math:[`k\lambda`]
        Baselines corresponding to `u` and `v`
    Vmod : array, unit=[Jy] 
        Visibility amplitudes at `q`

    Notes
    -----
    This routine requires the `frank <https://github.com/discsim/frank>`_ package # TODO
    """
    # from mpol.geometry import deproject_vis # TODO
    # deproject_vis(u, v, V, weights, source_geom, inverse, rescale_flux) # TODO

    q = model.coords.packed_q_centers_2D.ravel()
    Vmod = model.fcube.vis.detach()[chan].ravel()

    return q, Vmod


def get_radial_profile(model, center=(0.0, 0.0), bins=None, chan=0):
    r"""
    Obtain a 1D (radial) brightness profile I(r) from an MPoL model.

    Parameters
    ----------
    # TODO
    model : `torch.nn.Module` object
        Instance of the `mpol.precomposed.SimpleNet` class
    center : 2-tuple of float, default=(0.0, 0.0), unit=[arcsec]
        Offset (RA, Dec) of source in image. Postive RA offset is west of
        north. If None, the source is assumed to be at the image center pixel
    bins : array, default=None, unit=[arcsec]
        Radial bin edges to use in calculating I(r)
    chan : int, default=0
        Channel of `model` to select

    Returns
    -------
    bin_centers : array, unit=[arcsec]
        Radial coordinates of image at center of `bins`
    Is : array, unit=[Jy / arcsec^2]
        Azimuthally averaged pixel brightness at `rs`

    """
    # observer_to_flat(X, Y, omega, incl, Omega) # TODO

    skycube = torch2npy(model.icube.sky_cube)[chan]

    # Cartesian pixel coordinates [arcsec]
    xx, yy = model.coords.sky_x_centers_2D, model.coords.sky_y_centers_2D
    # shift image center
    xshift, yshift = xx - center[0], yy - center[1]
    # radial pixel coordinates
    rshift = np.hypot(xshift, yshift) 
    rshift = rshift.flatten()

    if bins is None:
        step = np.hypot(model.coords.cell_size, model.coords.cell_size)
        bins = np.arange(0.0, max(rshift), step)

    # get number of points in each radial bin
    bin_counts, bin_edges = np.histogram(a=rshift, bins=bins, weights=None)
    # get radial brightness
    Is, _ = np.histogram(a=rshift, bins=bins, weights=skycube.flatten())
    Is /= bin_counts
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, Is