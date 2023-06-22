import numpy as np 

from mpol.geometry import observer_to_flat
from mpol.utils import torch2npy


def get_1d_vis_profile(V, coords, geom, rescale_flux=True, bins=None):
    r"""
    Obtain the 1D (radial) visibility model V(q) corresponding to a 2D MPoL 
    image-domain model. 

    Parameters
    ----------
    V : array
        2D visibility amplitudes
    coords : `mpol.coordinates.GridCoords` object
        Instance of the `mpol.coordinates.GridCoords` class
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
    rescale_flux : bool, default=True 
        If True, the visibility amplitudes and weights are rescaled to account 
        for the difference between the inclined (observed) brightness and the 
        assumed face-on brightness, assuming the emission is optically thick. 
        The source's integrated (2D) flux is assumed to be:
            :math:`F = \cos(i) \int_r^{r=R}{I(r) 2 \pi r dr}`.
        No rescaling would be appropriate in the optically thin limit. 
    bins : array, default=None, unit=[k\lambda]
        Baseline bin edges to use in calculating V(q). If None, bins will span 
        the model baseline distribution, with widths equal to the hypotenuse of 
        the (u, v) coordinates

    Returns
    -------
    q : array, unit=:math:[`k\lambda`]
        Baselines corresponding to `u` and `v`
    Vs : array, unit=[Jy] 
        Visibility amplitudes at `q`

    Notes
    -----
    This routine requires the `frank <https://github.com/discsim/frank>`_ package
    """

    # model (u,v) coordinates [k\lambda]
    uu, vv = coords.sky_u_centers_2D, coords.sky_v_centers_2D

    from frank.geometry import apply_phase_shift, deproject
    # phase-shift the model visibilities
    Vp = apply_phase_shift(uu.ravel() * 1e3, vv.ravel() * 1e3, V.ravel(), geom["dRA"], geom["dDec"], inverse=True)
    # deproject the model (u,v) points
    up, vp, _ = deproject(uu.ravel() * 1e3, vv.ravel() * 1e3, geom["incl"], geom["Omega"])

    # if the source is optically thick, rescale the deprojected Re(V)
    if rescale_flux: 
        Vp.real /= np.cos(geom["incl"] * np.pi / 180)

    # convert back to [k\lambda]
    up /= 1e3
    vp /= 1e3
    qq = np.hypot(up, vp) 

    if bins is None:
        step = np.hypot(coords.du, coords.dv)
        bins = np.arange(0.0, max(qq), step)

    # number of points in radial bins
    bin_counts, bin_edges = np.histogram(a=qq, bins=bins, weights=None)
    # V amplitudes in radial bins
    Vs, _ = np.histogram(a=qq, bins=bins, weights=Vp)
    Vs /= bin_counts
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, Vs


def get_1d_image_profile(image, coords, geom, bins=None, rescale_flux=True): 
    r"""
    Obtain a 1D (radial) brightness profile I(r) from an image.

    Parameters
    ----------
    image : array
        2D image array 
    coords : `mpol.coordinates.GridCoords` object
        Instance of the `mpol.coordinates.GridCoords` class
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
    rescale_flux : bool, default=True 
        If True, the brightness values are rescaled to account for the 
        difference between the inclined (observed) brightness and the 
        assumed face-on brightness, assuming the emission is optically thick. 
        The source's integrated (2D) flux is assumed to be:
            :math:`F = \cos(i) \int_r^{r=R}{I(r) 2 \pi r dr}`.
        No rescaling would be appropriate in the optically thin limit. 
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

    # Cartesian pixel coordinates [arcsec]
    xx, yy = coords.sky_x_centers_2D, coords.sky_y_centers_2D
    # shift image center
    xshift, yshift = xx - geom["dRA"], yy - geom["dDec"]

    # deproject and rotate image 
    xdep, ydep = observer_to_flat(xshift, yshift,
        omega=geom["omega"] * np.pi / 180, 
        incl=geom["incl"] * np.pi / 180,
        Omega=geom["Omega"] * np.pi / 180,
        )
    
    # radial pixel coordinates
    rr = np.ravel(np.hypot(xdep, ydep))

    if bins is None:
        step = np.hypot(coords.cell_size, coords.cell_size)
        bins = np.arange(0.0, max(rr), step)

    # number of points in radial bins
    bin_counts, bin_edges = np.histogram(a=rr, bins=bins, weights=None)
    # brightness in radial bins
    Is, _ = np.histogram(a=rr, bins=bins, weights=np.ravel(image))
    Is /= bin_counts

    if rescale_flux: # TODO
        Is *= np.cos(geom["incl"] * np.pi / 180)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, Is
