"""The geometry package provides routines for projecting and de-projecting sky images and visibilities.
"""

import numpy as np
import torch


def flat_to_observer(x, y, omega=None, incl=None, Omega=None):
    """Rotate the frame to convert a point in the flat (x,y,z) frame to observer frame (X,Y,Z).

    It is assumed that the +Z axis points *towards* the observer. It is assumed that the model is flat in the (x,y) frame (like a flat disk), and so the operations involving ``z`` are neglected. But the model lives in 3D Cartesian space.

    In order,

    1. rotate about the z axis by an amount omega -> x1, y1, z1
    2. rotate about the x1 axis by an amount -incl -> x2, y2, z2
    3. rotate about the z2 axis by an amount Omega -> x3, y3, z3 = X, Y, Z

    Inspired by `exoplanet/keplerian.py <https://github.com/exoplanet-dev/exoplanet/blob/main/src/exoplanet/orbits/keplerian.py>`_

    Args:
        x (torch tensor): A tensor representing the x coordinate in the plane of the orbit.
        y (torch tensor): A tensor representing the y coordinate in the plane of the orbit.
        omega (torch float tensor): A tensor representing an argument of periastron [radians] Default 0.0.
        incl (torch float tensor): A tensor representing an inclination value [radians]. Default 0.0.
        Omega (torch float tensor): A tensor representing the position angle of the ascending node in [radians]. Default 0.0

    Returns:
        Two tensors representing ``(X, Y)`` in the observer frame.
    """

    # Rotation matrices result in a *clockwise* rotation of the axes, as defined using the righthand rule.
    # For example, looking down the z-axis, a positive angle will rotate the x,y axes clockwise.
    # A vector in the coordinate system will appear as though it has been rotated counter-clockwise.

    # 1) rotate about the z0 axis by omega
    if omega is not None:
        cos_omega = torch.cos(torch.as_tensor(omega))
        sin_omega = torch.sin(torch.as_tensor(omega))

        x1 = cos_omega * x - sin_omega * y
        y1 = sin_omega * x + cos_omega * y
    else:
        x1 = x
        y1 = y

    # 2) rotate about x1 axis by -incl
    x2 = x1

    if incl is not None:
        y2 = torch.cos(torch.as_tensor(incl)) * y1
        # z3 = z2, subsequent rotation by Omega doesn't affect it
        # Z = -torch.sin(incl) * y1
    else:
        y2 = y1
        # Z = 0.0

    # 3) rotate about z2 axis by Omega
    if Omega is not None:
        cos_Omega = torch.cos(torch.as_tensor(Omega))
        sin_Omega = torch.sin(torch.as_tensor(Omega))

        X = cos_Omega * x2 - sin_Omega * y2
        Y = sin_Omega * x2 + cos_Omega * y2
    else:
        X = x2
        Y = y2

    return X, Y


def observer_to_flat(X, Y, omega=None, incl=None, Omega=None):
    """Rotate the frame to convert a point in the observer frame (X,Y,Z) to the flat (x,y,z) frame.

    It is assumed that the +Z axis points *towards* the observer. The rotation operations are the inverse of the :func:`~mpol.geometry.flat_to_observer` operations.

    In order,

    1. inverse rotation about the Z axis by an amount Omega -> x2, y2, z2
    2. inverse rotation about the x2 axis by an amount -incl -> x1, y1, z1
    3. inverse rotation about the z1 axis by an amount omega -> x, y, z

    Inspired by `exoplanet/keplerian.py <https://github.com/exoplanet-dev/exoplanet/blob/main/src/exoplanet/orbits/keplerian.py>`_

    Args:
        X (torch tensor): A tensor representing the x coodinate in the plane of the orbit.
        Y (torch.tensor): A tensor representing the y coodinate in the plane of the orbit.
        omega (torch float tensor): A tensor representing an argument of periastron [radians] Default 0.0.
        incl (torch float tensor): A tensor representing an inclination value [radians]. Default 0.0.
        Omega (torch float tensor): A tensor representing the position angle of the ascending node in [radians]. Default 0.0

    Returns:
        Two tensors representing ``(x, y)`` in the flat frame.
    """

    # Rotation matrices result in a *clockwise* rotation of the axes, as defined using the righthand rule.
    # For example, looking down the z-axis, a positive angle will rotate the x,y axes clockwise.
    # A vector in the coordinate system will appear as though it has been rotated counter-clockwise.

    # 1) inverse rotation about Z axis by Omega -> x2, y2, z2
    if Omega is not None:
        cos_Omega = torch.cos(torch.as_tensor(Omega))
        sin_Omega = torch.sin(torch.as_tensor(Omega))

        x2 = cos_Omega * X + sin_Omega * Y
        y2 = -sin_Omega * X + cos_Omega * Y
    else:
        x2 = X
        y2 = Y

    # 2) inverse rotation about x2 axis by incl
    x1 = x2
    # we don't know Z, but we can solve some equations to find that
    # y = Y / cos(i), as expected by intuition
    if incl is not None:
        y1 = y2 / torch.cos(torch.as_tensor(incl))
    else:
        y1 = y2

    # 3) inverse rotation about the z1 axis by an amount of omega
    if omega is not None:
        cos_omega = torch.cos(torch.as_tensor(omega))
        sin_omega = torch.sin(torch.as_tensor(omega))

        x = x1 * cos_omega + y1 * sin_omega
        y = -x1 * sin_omega + y1 * cos_omega
    else:
        x = x1
        y = y1

    return x, y


# TODO
def deproject_vis(u, v, V, weights, source_geom, inverse=False, rescale_flux=True): 
    r"""
    Deproject (or reproject) visibilities (and optionally rescale their 
    amplitudes and weights, according to the source inclination) using routines 
    in `frank <https://github.com/discsim/frank>`_

    Parameters
    ----------
    u : array of real, size = N, unit = :math:`\lambda`
        u-points of the visibilities
    v : array of real, size = N, unit = :math:`\lambda`
        v-points of the visibilities
    V : array of real, size = N, unit = Jy
        Complex visibilites
    weights : array of real, size = N, unit = Jy
        Weights on the visibilities    
    source_geom : dict
        Dictionary with source geometry parameters. Keys:
            "inc" : float, unit = deg
                Inclination
            "PA" : float, unit = deg
                Position angle, defined east of north.  
            "dRA" : float, unit = arcsec
                Phase centre offset in right ascension.
            "dDec" : float, units = arcsec
                Phase centre offset in declination.           
    inverse : bool, default=False
        If True, the uv-points are reprojected rather than deprojected
    rescale_flux : bool, default=True
        If True, the visibility amplitudes and weights are rescaled to account 
        for the difference between the inclined (observed) brightness and the 
        assumed face-on brightness, assuming the emission is optically thick. 
        The source's integrated (2D) flux is assumed to be:
            :math:`F = \cos(i) \int_r^{r=R}{I(r) 2 \pi r dr}`.
        No rescaling would be appropriate in the optically thin limit.
    Returns
    -------
    up : array of real, size = N, unit = :math:`\lambda`
        Corrected u-points of the visibilities
    vp : array of real, size = N, unit = :math:`\lambda`
        Corrected v-points of the visibilities
    Vp : array of real, size = N, unit = Jy
        Corrected complex visibilites
    weights_scaled : array of real, size = N, unit = Jy
        Rescaled weights on the visibilities        

    """
    from frank.geometry import FixedGeometry

    geom = FixedGeometry(**source_geom)

    weights_scaled = weights * 1

    if inverse:
        up, vp, Vp = geom.apply_correction(u, v, V)
        if rescale_flux:
            Vp, weights_scaled = geom.rescale_total_flux(V, weights)

    else:
        if rescale_flux:
            Vp, weights_scaled = geom.rescale_total_flux(V, weights)
         up, vp, Vp = geom.undo_correction(u, v, V)

    return up, vp, Vp, weights_scaled
