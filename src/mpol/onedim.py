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

    from frank.geometry import FixedGeometry
    geom_frank = FixedGeometry(geom["incl"], geom["Omega"], geom["dRA"], geom["dDec"])  # TODO: signs
    # phase-shift the model visibilities and deproject the model (u,v) points
    up, vp, Vp = geom_frank.apply_correction(uu.ravel() * 1e3, vv.ravel() * 1e3, V.ravel())

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

    # get number of points in each radial bin
    bin_counts, bin_edges = np.histogram(a=qq, bins=bins, weights=None)
    # get radial vis amplitudes
    Vs, _ = np.histogram(a=qq, bins=bins, weights=Vp)
    Vs /= bin_counts
    
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, Vs


def get_radial_profile(model, geom, bins=None, rescale_flux=True, chan=0): 
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

    # Cartesian pixel coordinates [arcsec]
    xx, yy = model.coords.sky_x_centers_2D, model.coords.sky_y_centers_2D
    # shift image center
    xshift, yshift = xx - geom["dRA"], yy - geom["dDec"]

    # deproject and rotate image 
    xdep, ydep = observer_to_flat(xshift, yshift,
        omega=geom["omega"] * np.pi / 180, # TODO: omega
        incl=geom["incl"] * np.pi / 180,
        Omega=geom["Omega"] * np.pi / 180,
        opt_thick=rescale_flux,
        )

    # radial pixel coordinates
    rr = np.ravel(np.hypot(xdep, ydep))

    if bins is None:
        step = np.hypot(model.coords.cell_size, model.coords.cell_size)
        bins = np.arange(0.0, max(rr), step)

    # get number of points in each radial bin
    bin_counts, bin_edges = np.histogram(a=rr, bins=bins, weights=None)
    # get radial brightness
    Is, _ = np.histogram(a=rr, bins=bins, weights=np.ravel(skycube))
    Is /= bin_counts

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centers, Is
