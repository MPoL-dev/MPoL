import numpy as np 

from mpol.fourier import NuFFT
from mpol.geometry import observer_to_flat
from mpol.utils import torch2npy


def get_1d_vis_fit(model, geom, bins=None, rescale_flux=True, chan=0):
    r"""
    Obtain the 1D (radial) visibility model V(q) corresponding to a 2D MPoL 
    image-domain model. 

    Parameters
    ----------
    model : `torch.nn.Module` object
        Instance of the `mpol.precomposed.SimpleNet` class
    geom : dict 
        Dictionary of source geometry. Keys:
            "incl" : float, unit=[deg]
                Inclination 
            "Omega" : float, unit=[deg]
                Position angle of the ascending node # TODO: convention?
            "omega" : float, unit=[deg]
                Argument of periastron
            "dRA" : float, unit=[arcsec]
                Phase center offset in right ascension. Positive is west of north. # TODO: convention?
            "dDec" : float, unit=[arcsec]
                Phase center offset in declination
    rescale_flux : bool, default=True # TODO
        If True, the visibility amplitudes and weights are rescaled to account 
        for the difference between the inclined (observed) brightness and the 
        assumed face-on brightness, assuming the emission is optically thick. 
        The source's integrated (2D) flux is assumed to be:
            :math:`F = \cos(i) \int_r^{r=R}{I(r) 2 \pi r dr}`.
        No rescaling would be appropriate in the optically thin limit.                
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
    This routine requires the `frank <https://github.com/discsim/frank>`_ package
    """

    # model visibility amplitudes
    Vmod = torch2npy(model.fcube.ground_vis)[chan] # TODO: or is it model.fcube.vis?

    # model (u,v) coordinates [k\lambda]
    uu, vv = model.coords.sky_u_centers_2D, model.coords.sky_v_centers_2D

    from frank.geometry import FixedGeometry
    geom_frank = FixedGeometry(geom["incl"], geom["Omega"], geom["dRA"], geom["dDec"])    
    # phase-shift the model visibilities and deproject the model (u,v) points
    up, vp, Vp = geom_frank.apply_correction(uu.ravel() * 1e3, vv.ravel() * 1e3, Vmod.ravel())
    # if rescale_flux: # TODO: be consistent w/ get_radial_profile
    #     Vp, weights_scaled = geom_frank.rescale_total_flux(Vp, weights)
    # convert back to [k\lambda]
    up /= 1e3
    vp /= 1e3

    # baselines
    qq = np.hypot(up, vp) 

    if bins is None:
        step = np.hypot(model.coords.du, model.coords.dv)
        bins = np.arange(0.0, max(qq), step)

    # get number of points in each radial bin
    bin_counts, bin_edges = np.histogram(a=qq, bins=bins, weights=None)
    # get radial vis amplitude
    Vs, _ = np.histogram(a=qq, bins=bins, weights=Vp)
    Vs /= bin_counts
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, Vs


def get_radial_profile(model, geom, bins=None, chan=0):
    r"""
    Obtain a 1D (radial) brightness profile I(r) from an MPoL model.

    Parameters
    ----------
    model : `torch.nn.Module` object
        Instance of the `mpol.precomposed.SimpleNet` class
    geom : dict 
        Dictionary of source geometry. Keys:
            "incl" : float, unit=[deg]
                Inclination 
            "Omega" : float, unit=[deg]
                Position angle of the ascending node # TODO: convention?
            "omega" : float, unit=[deg]
                Argument of periastron
            "dRA" : float, unit=[arcsec]
                Phase center offset in right ascension. Positive is west of north. # TODO: convention?
            "dDec" : float, unit=[arcsec]
                Phase center offset in declination.
    bins : array, default=None, unit=[arcsec]
        Radial bin edges to use in calculating I(r). If None, bins will span 
        the full image, with widths equal to the hypotenuse of the pixels
    chan : int, default=0
        Channel of `model` to use

    Returns
    -------
    bin_centers : array, unit=[arcsec]
        Radial coordinates of image at center of `bins`
    Is : array, unit=[Jy / arcsec^2]
        Azimuthally averaged pixel brightness at `rs`

    """

    # model pixel values
    skycube = torch2npy(model.icube.sky_cube)[chan]
    # TODO: scale (multiply) brightness by inclination? if so, add arg

    # Cartesian pixel coordinates [arcsec]
    xx, yy = model.coords.sky_x_centers_2D, model.coords.sky_y_centers_2D
    # shift image center
    xshift, yshift = xx - geom["dRA"], yy - geom["dDec"]

    # deproject and rotate image 
    xdep, ydep = observer_to_flat(xshift, yshift,
        omega=geom["omega"] * np.pi / 180, # TODO: omega
        incl=geom["incl"] * np.pi / 180,
        Omega=geom["Omega"] * np.pi / 180)

    # radial pixel coordinates
    rr = np.hypot(xdep, ydep) 
    rr = rr.flatten()

    if bins is None:
        step = np.hypot(model.coords.cell_size, model.coords.cell_size)
        bins = np.arange(0.0, max(rr), step)

    # get number of points in each radial bin
    bin_counts, bin_edges = np.histogram(a=rr, bins=bins, weights=None)
    # get radial brightness
    Is, _ = np.histogram(a=rr, bins=bins, weights=skycube.flatten())
    Is /= bin_counts
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, Is
