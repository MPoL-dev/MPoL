import numpy as np 
from mpol.utils import torch2npy

def radialI(icube, geom, chan=0, bins=None):
    r"""
    Obtain a 1D (radial) brightness profile I(r) from an image cube.

    Parameters
    ----------
    icube : `mpol.images.ImageCube` object
        Instance of the MPoL `images.ImageCube` class
    geom : dict 
        Dictionary of source geometry. Keys:
            "incl" : float, unit=[deg]
                Inclination 
            "Omega" : float, unit=[deg]
                Position angle of the ascending node 
            "omega" : float, unit=[deg]
                Argument of periastron
            "dRA" : float, unit=[arcsec]
                Phase center offset in right ascension. Positive is west of north.
            "dDec" : float, unit=[arcsec]
                Phase center offset in declination.
    chan : int, default=0
        Channel of the image cube corresponding to the desired image
    bins : array, default=None, unit=[arcsec]
        Radial bin edges to use in calculating I(r). If None, bins will span 
        the full image, with widths equal to the hypotenuse of the pixels

    Returns
    -------
    bin_centers : array, unit=[arcsec]
        Radial coordinates of image at center of `bins`
    Is : array, unit=[Jy / arcsec^2] (if `image` has these units)
        Azimuthally averaged pixel brightness at `rs`
    """

    # projected Cartesian pixel coordinates [arcsec]
    xx, yy = icube.coords.sky_x_centers_2D, icube.coords.sky_y_centers_2D

    # shift image center to source center
    xc, yc = xx - geom["dRA"], yy - geom["dDec"]

    # deproject image
    cos_PA = np.cos(geom["Omega"] * np.pi / 180)
    sin_PA = np.sin(geom["Omega"] * np.pi / 180)
    xd = xc * cos_PA - yc * sin_PA
    yd = xc * sin_PA + yc * cos_PA
    xd /= np.cos(geom["incl"] * np.pi / 180)

    # deprojected radial coordinates
    rr = np.ravel(np.hypot(xd, yd))

    if bins is None:
        # choose sensible bin size and range
        step = np.hypot(icube.coords.cell_size, icube.coords.cell_size)
        bins = np.arange(0.0, np.max((abs(xc.ravel()), abs(yc.ravel()))), step)

    bin_counts, bin_edges = np.histogram(a=rr, bins=bins, weights=None)

    # cumulative binned brightness in each annulus
    Is, _ = np.histogram(a=rr, bins=bins, 
                         weights=torch2npy(icube.sky_cube[chan]).ravel()
                         )

    # mask empty bins
    mask = (bin_counts == 0)
    Is = np.ma.masked_where(mask, Is)

    # average binned brightness in each annulus
    Is /= bin_counts

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, Is


def radialV(fcube, geom, rescale_flux, chan=0, bins=None):
    r"""
    Obtain the 1D (radial) visibility model V(q) corresponding to a 2D MPoL 
    image. 

    Parameters
    ----------
    fcube : `~mpol.fourier.FourierCube` object
        Instance of the MPoL `fourier.FourierCube` class
    geom : dict 
        Dictionary of source geometry. Keys:
            "incl" : float, unit=[deg]
                Inclination 
            "Omega" : float, unit=[deg]
                Position angle of the ascending node 
            "omega" : float, unit=[deg]
                Argument of periastron
            "dRA" : float, unit=[arcsec]
                Phase center offset in right ascension. Positive is west of north. 
            "dDec" : float, unit=[arcsec]
                Phase center offset in declination
    rescale_flux : bool
        If True, the visibility amplitudes and weights are rescaled to account 
        for the difference between the inclined (observed) brightness and the 
        assumed face-on brightness, assuming the emission is optically thick. 
        The source's integrated (2D) flux is assumed to be:
            :math:`F = \cos(i) \int_r^{r=R}{I(r) 2 \pi r dr}`.
        No rescaling would be appropriate in the optically thin limit. 
    chan : int, default=0
        Channel of the image cube corresponding to the desired image        
    bins : array, default=None, unit=[k\lambda]
        Baseline bin edges to use in calculating V(q). If None, bins will span 
        the model baseline distribution, with widths equal to the hypotenuse of 
        the (u, v) coordinates

    Returns
    -------
    bin_centers : array, unit=:math:[`k\lambda`]
        Baselines corresponding to `u` and `v`
    Vs : array, unit=[Jy] 
        Visibility amplitudes at `q`

    Notes
    -----
    This routine requires the `frank <https://github.com/discsim/frank>`_ package
    """
    from frank.geometry import apply_phase_shift, deproject

    # projected model (u,v) points [k\lambda]
    uu, vv = fcube.coords.sky_u_centers_2D, fcube.coords.sky_v_centers_2D

    # visibilities 
    V = torch2npy(fcube.ground_cube[chan]).ravel()

    # phase-shift the visibilities
    Vp = apply_phase_shift(uu.ravel() * 1e3, vv.ravel() * 1e3, V, geom["dRA"], 
                           geom["dDec"], inverse=True)
    
    # deproject the (u,v) points
    up, vp, _ = deproject(uu.ravel() * 1e3, vv.ravel() * 1e3, geom["incl"], 
                          geom["Omega"])

    # if the source is optically thick, rescale the deprojected V(q)
    if rescale_flux: 
        Vp.real /= np.cos(geom["incl"] * np.pi / 180)

    # convert back to [k\lambda]
    up /= 1e3
    vp /= 1e3

    # deprojected baselines
    qq = np.hypot(up, vp) 

    if bins is None:
        # choose sensible bin size and range
        step = np.hypot(fcube.coords.du, fcube.coords.dv) / 2
        bins = np.arange(0.0, max(qq), step)

    bin_counts, bin_edges = np.histogram(a=qq, bins=bins, weights=None)

    # cumulative binned visibility amplitude in each annulus
    Vs, _ = np.histogram(a=qq, bins=bins, weights=Vp)

    # mask empty bins
    mask = (bin_counts == 0)
    Vs = np.ma.masked_where(mask, Vs)

    # average binned visibility amplitude in each annulus
    Vs /= bin_counts
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, Vs