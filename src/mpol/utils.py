import math
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from mpol.constants import arcsec, cc, deg, kB


def torch2npy(t: torch.Tensor) -> npt.NDArray:
    """
    Copy a tensor (potentially on the GPU) to the CPU and convert to a numpy
    :class:`np.ndarray`, e.g., for visualization or further analysis with non-PyTorch
    scientific libraries.

    Parameters
    ----------
    t : torch.Tensor

    Returns
    _______
    np.ndarray

    """
    t_cpu: torch.Tensor = t.detach().cpu()
    t_np: np.ndarray = t_cpu.numpy()
    return t_np


def ground_cube_to_packed_cube(ground_cube: torch.Tensor) -> torch.Tensor:
    r"""
    Converts a Ground Cube to a Packed Visibility Cube for visibility-plane work.
    See Units and Conventions for more details.

    Args:
        ground_cube: a previously initialized Ground Cube object (cube (3D torch tensor
        of shape ``(nchan, npix, npix)``))

    Returns:
        torch.double : 3D image cube of shape ``(nchan, npix, npix)``; The resulting
            array after applying ``torch.fft.fftshift`` to the input arg; i.e Returns a
            Packed Visibility Cube.
    """
    shifted: torch.Tensor = torch.fft.fftshift(ground_cube, dim=(1, 2))
    return shifted


def packed_cube_to_ground_cube(packed_cube: torch.Tensor) -> torch.Tensor:
    r"""
    Converts a Packed Visibility Cube to a Ground Cube for visibility-plane work. See
    Units and Conventions for more details.

    Args:
        packed_cube: a previously initialized Packed Cube object (cube (3D torch tensor
        of shape ``(nchan, npix, npix)``))

    Returns:
        torch.double : 3D image cube of shape ``(nchan, npix, npix)``; The resulting
            array after applying ``torch.fft.fftshift`` to the input arg; i.e Returns a
            Ground Cube.
    """
    # fftshift the image cube to the correct quadrants
    shifted: torch.Tensor = torch.fft.fftshift(packed_cube, dim=(1, 2))
    return shifted


def sky_cube_to_packed_cube(sky_cube: torch.Tensor) -> torch.Tensor:
    r"""
    Converts a Sky Cube to a Packed Image Cube for image-plane work. See Units and
    Conventions for more details.

    Args:
        sky_cube: a previously initialized Sky Cube object with RA increasing to the
            *left* (cube (3D torch tensor of shape ``(nchan, npix, npix)``))

    Returns:
        torch.double : 3D image cube of shape ``(nchan, npix, npix)``; The resulting
            array after applying ``torch.fft.fftshift`` to the ``torch.flip()`` of the
            RA axis; i.e Returns a Packed Image Cube.
    """
    flipped = torch.flip(sky_cube, (2,))
    shifted: torch.Tensor = torch.fft.fftshift(flipped, dim=(1, 2))
    return shifted


def packed_cube_to_sky_cube(packed_cube: torch.Tensor) -> torch.Tensor:
    r"""
    Converts a Packed Image Cube to a Sky Cube for image-plane work. See Units and
    Conventions for more details.

    Args:
        packed_cube: a previously initialized Packed Image Cube object (cube (3D torch
        tensor of shape ``(nchan, npix, npix)``))

    Returns:
        torch.double : 3D image cube of shape ``(nchan, npix, npix)``; The resulting
            array after applying ``torch.fft.fftshift`` to the ``torch.flip()`` of the
            RA axis; i.e Returns a Sky Cube.
    """
    # fftshift the image cube to the correct quadrants
    shifted = torch.fft.fftshift(packed_cube, dim=(1, 2))
    # flip so that east points left
    flipped = torch.flip(shifted, (2,))
    return flipped


def get_Jy_arcsec2(T_b: float, nu: float = 230e9) -> float:
    r"""
    Calculate specific intensity from the brightness temperature, using the
    Rayleigh-Jeans definition.

    Args:
        T_b : brightness temperature in [:math:`K`]
        nu : frequency (in Hz)

    Returns:
        float: specific intensity (in [:math:`\mathrm{Jy}\, \mathrm{arcsec}^2]`)
    """
    # brightness temperature assuming RJ limit
    # units of ergs/s/cm^2/Hz/ster
    I_nu = T_b * 2 * nu**2 * kB / cc**2

    # convert to Jy/ster
    Jy_ster = I_nu * 1e23

    # convert to Jy/arcsec^2
    Jy_arcsec2 = Jy_ster * arcsec**2

    return Jy_arcsec2


def loglinspace(
    start: float, end: float, N_log: int, M_linear: int = 3
) -> npt.NDArray[np.floating[Any]]:
    r"""
    Return a logspaced array of bin edges, with the first ``M_linear`` cells being
    equal width. There is a one-cell overlap between the linear and logarithmic
    stretches of the array, since the last linear cell is also the first logarithmic
    cell, which means the total number of cells is ``M_linear + N_log - 1``.

    Parameters
    ----------
    start : float
        starting cell left edge
    end : float
        ending cell right edge
    N_log : int
        number of logarithmically spaced bins
    M_linear : int
        number of linearly (equally) spaced bins

    Returns
    -------
    np.ndarray
        logspaced bin edges

    """

    # transition cell left edge
    a = end / 10 ** (N_log * np.log10(M_linear / (M_linear - 1)))
    delta = a / (M_linear - 1)  # linear cell width

    # logspace = 10^(log10(a) + i * Delta)
    Delta = np.log10(end / a) / N_log  # log cell width exponent

    cell_walls = []
    for i in range(M_linear):
        cell_walls.append(start + delta * i)

    for j in range(1, N_log + 1):
        cell_walls.append(10 ** (np.log10(a) + Delta * j))

    return np.array(cell_walls)


def fftspace(width: float, N: int) -> npt.NDArray[np.floating[Any]]:
    """Delivers a (nearly) symmetric coordinate array that spans :math:`N` elements
    (where :math:`N` is even) from `-width` to `+width`, but ensures that the middle
    point lands on :math:`0`. The array indices go from :math:`0` to :math:`N -1.`

    Args:
        width (float): the width of the array
        N (int): the number of elements in the array

    Returns:
        numpy.float64 1D array: the fftspace array

    """
    assert N % 2 == 0, "N must be even."

    dx = width * 2.0 / N
    xx = np.empty(N, "float")
    for i in range(N):
        xx[i] = -width + i * dx

    return xx


def check_baselines(q, min_feasible_q=1e3, max_feasible_q=1e8):
    r"""
    Check if baseline lengths are sensible for expected code unit of
    [:math:`\lambda`], or if instead they're being supplied in
    [:math:`\mathrm{k}\lambda`].

    Parameters
    ----------
     q : array, unit = :math:`\lambda`
        Baseline distribution (all values must be non-negative).
    min_feasible_q : float, unit = :math:`\lambda`, default=1e3
        Minimum baseline in code units expected for a dataset. The default
        value of 1e3 is a conservative value for ALMA, assuming a minimum
        antenna separation of ~12 m and maximum observing wavelength of 3.6 mm.
    max_feasible_q : float, unit = :math:`\lambda`, default=1e8
        Maximum baseline in code units expected for a dataset. The default
        value of 1e8 is a conservative value for ALMA, assuming a maximum
        antenna separation of ~16 km and minimum observing wavelength of 0.3 mm.
    """

    assert np.all(q >= 0), "All baselines should be >=0."

    if max(q) > max_feasible_q:
        raise Warning(
            f"Maximum baseline of {max(q):.1e} is > maximum expected "
            f"value of {max_feasible_q:.1e}. Baselines must be in units of "
            "[klambda], but it looks like they're in "
            "[lambda]."
        )

    if min(q) > min_feasible_q * 1e3:
        raise Warning(
            "Minimum baseline of {:.1e} is large for expected "
            "minimum value of {:.1e}. Baselines must be in units of "
            "[lambda]".format(min(q), min_feasible_q * 1e3)
        )


def get_max_spatial_freq(cell_size: float, npix: int) -> float:
    r"""
    Calculate the maximum spatial frequency that the image can represent and still
    satisfy the Nyquist Sampling theorem.

    Parameters
    ----------
    cell_size : float
        the pixel size in arcseconds
    npix : int
        the number of pixels in the image

    Returns
    -------
    float
        the maximum spatial frequency contained in the image [:math:`\lambda`]
    """

    # technically this is as straightforward as doing 1/(2 * cell_size), but for even-sized
    # arrays, the highest *positive* spatial frequency is (npix/2 - 1) / (npix * cell_size)
    # it is the most negative spatial frequency that goes to - 1/(2 * cell_size)

    return (npix / 2 - 1) / (npix * cell_size * arcsec)  # λ


def get_maximum_cell_size(uu_vv_point: float) -> float:
    r"""
    Calculate the maximum possible cell_size that will still Nyquist sample the uu or
    vv point. Note: not q point.

    Args:
        uu_vv_point (float): a single spatial frequency. Units of
            [:math:`\lambda`].

    Returns:
        cell_size (in arcsec)
    """

    return 1 / ((2 - 1) * uu_vv_point) / arcsec


def get_optimal_image_properties(
    image_width: float,
    u: torch.Tensor,
    v: torch.Tensor,
) -> tuple[float, int]:
    r"""
    For an image of desired width, determine the maximum pixel size that
    ensures Nyquist sampling of the provided spatial frequency points, and the
    corresponding number of pixels to obtain the desired image width.

    Parameters
    ----------
    image_width : float, unit = arcsec
        Desired width of the image (for a square image of size
        `image_width` :math:`\times` `image_width`).
    u, v : :class:`torch.Tensor` , unit = :math:`\lambda`
        `u` and `v` baselines.

    Returns
    -------
    cell_size : float, unit = arcsec
        Image pixel size required to Nyquist sample.
    npix : int
        Number of pixels of cell_size to equal (or slightly exceed) the image
        width (npix will be rounded up and enforced as even).
    Notes
    -----
    Assumes baselines are as-observed.
    """
    max_freq = torch.max(torch.abs(torch.concat([u, v])))

    cell_size = get_maximum_cell_size(max_freq.item())

    # round npix up to nearest integer
    npix = math.ceil(image_width / cell_size)

    # account for Nyquist of proposed cell_size, npix
    cell_size *= cell_size / get_maximum_cell_size(
        get_max_spatial_freq(cell_size, npix)
    )

    npix = math.ceil(image_width / cell_size)

    # enforce that npix be even
    if npix % 2 == 1:
        npix += 1

    # should never occur
    assert (
        get_max_spatial_freq(cell_size, npix) >= max_freq
    ), "error in get_optimal_image_properties"

    return cell_size, npix


def sky_gaussian_radians(
    l: npt.NDArray[np.floating[Any]],
    m: npt.NDArray[np.floating[Any]],
    a: float,
    delta_l: float,
    delta_m: float,
    sigma_l: float,
    sigma_m: float,
    Omega: float,
) -> npt.NDArray[np.floating[Any]]:
    r"""
    Calculates a 2D Gaussian on the sky plane with inputs in radians. The Gaussian is
    centered at ``delta_l, delta_m``, has widths of ``sigma_l, sigma_m``, and is
    rotated ``Omega`` degrees East of North.

    To evaluate the Gaussian, internally first we translate to center

    .. math::

        l' = l - \delta_l\\
        m' = m - \delta_m

    then rotate coordinates

    .. math::

        l'' = l' \cos \phi - m' \sin \phi \\
        m'' = l' \sin \phi + m' \cos \phi

    and then evaluate the Gaussian

    .. math::

        f_\mathrm{g}(l,m) = a \exp \left ( - \frac{1}{2} \left
        [ \left (\frac{l''}{\sigma_l} \right)^2 + \left( \frac{m''}{\sigma_m}
        \right )^2 \right ] \right )

    Args:
        l: units of [radians]
        m: units of [radians]
        a : amplitude prefactor
        delta_l : offset [radians]
        delta_m : offset [radians]
        sigma_l : width [radians]
        sigma_m : width [radians]
        Omega : position angle of ascending node [degrees] east of north.

    Returns:
        2D Gaussian evaluated at input args with peak amplitude :math:`a`
    """

    # translate
    lt = l - delta_l
    mt = m - delta_m

    # rotate
    lp = lt * np.cos(Omega * deg) - mt * np.sin(Omega * deg)
    mp = lt * np.sin(Omega * deg) + mt * np.cos(Omega * deg)

    gauss: npt.NDArray[np.floating[Any]] = a * np.exp(
        -0.5 * ((lp / sigma_l) ** 2 + (mp / sigma_m) ** 2)
    )
    return gauss


def sky_gaussian_arcsec(
    x: npt.NDArray[np.floating[Any]],
    y: npt.NDArray[np.floating[Any]],
    a: float,
    delta_x: float,
    delta_y: float,
    sigma_x: float,
    sigma_y: float,
    Omega: float,
) -> npt.NDArray[np.floating[Any]]:
    r"""
    Calculates a Gaussian on the sky plane using inputs in arcsec. This is a convenience
    wrapper to :func:`~mpol.utils.sky_gaussian_radians` that automatically converts
    from arcsec to radians.

    Args:
        x: equivalent to l, but in units of [arcsec]
        y: equivalent to m, but in units of [arcsec]
        a : amplitude prefactor
        delta_x : offset [arcsec]
        delta_y : offset [arcsec]
        sigma_x : width [arcsec]
        sigma_y : width [arcsec]
        Omega : position angle of ascending node [degrees] east of north.

    Returns:
        2D Gaussian evaluated at input args with peak amplitude :math:`a`
    """

    return sky_gaussian_radians(
        x * arcsec,
        y * arcsec,
        a,
        delta_x * arcsec,
        delta_y * arcsec,
        sigma_x * arcsec,
        sigma_y * arcsec,
        Omega,
    )


def fourier_gaussian_lambda_radians(
    u: npt.NDArray[np.floating[Any]],
    v: npt.NDArray[np.floating[Any]],
    a: float,
    delta_l: float,
    delta_m: float,
    sigma_l: float,
    sigma_m: float,
    Omega: float,
) -> npt.NDArray[np.floating[Any]]:
    r"""
    Calculate the Fourier plane Gaussian :math:`F_\mathrm{g}(u,v)` corresponding to the
    Sky plane Gaussian :math:`f_\mathrm{g}(l,m)` in
    :func:`~mpol.utils.sky_gaussian_radians`, using analytical relationships. The
    Fourier Gaussian is parameterized using the sky plane centroid
    (``delta_l, delta_m``), widths (``sigma_l, sigma_m``) and rotation (``Omega``).
    Assumes that ``a`` was in units of :math:`\mathrm{Jy}/\mathrm{steradian}`.

    Args:
        u: l in units of [lambda]
        v: m in units of [lambda]
        a : amplitude prefactor, units of :math:`\mathrm{Jy}/\mathrm{steradian}`.
        delta_x : offset [radians]
        delta_y : offset [radians]
        sigma_x : width [radians]
        sigma_y : width [radians]
        Omega : position angle of ascending node [degrees] east of north.

    Returns:
        2D Gaussian evaluated at input args

    The following is a description of how we derived the analytical relationships. In
    what follows, all :math:`l` and :math:`m` coordinates are assumed to be in units
    of radians and all :math:`u` and :math:`v` coordinates are assumed to be in units
    of :math:`\lambda`.

    We start from Fourier dual relationships in Bracewell's `The Fourier Transform and
    Its Applications <https://ui.adsabs.harvard.edu/abs/2000fta..book.....B/abstract>`_

    .. math::

        f_0(l, m) \leftrightharpoons F_0(u, v)

    where the sky-plane and Fourier-plane Gaussians are

    .. math::

        f_0(l,m) = a \exp \left ( -\pi [l^2 + m^2] \right)

    and

    .. math::

        F_0(u,v) = a \exp \left ( -\pi [u^2 + v^2] \right),

    respectively. The sky-plane Gaussian has a maximum value of :math:`a`.

    We will use the similarity, rotation, and shift theorems to turn :math:`f_0` into
    a form matching :math:`f_\mathrm{g}`, which simultaneously turns :math:`F_0` into
    :math:`F_\mathrm{g}(u,v)`.

    The similarity theorem states that (in 1D)

    .. math::

        f(bl) = \frac{1}{|b|}F\left(\frac{u}{b}\right).

    First, we scale :math:`f_0` to include sigmas. Let

    .. math::

        f_1(l, m) = a \exp \left(-\frac{1}{2} \left [\left(\frac{l}{\sigma_l}\right)^2 +
        \left( \frac{m}{\sigma_m} \right)^2 \right] \right).

    i.e., something we might call a normalized Gaussian function. Phrased in terms of
    :math:`f_0`, :math:`f_1` is

    .. math::

        f_1(l, m) = f_0\left ( \frac{l}{\sigma_l \sqrt{2 \pi}},\,
        \frac{m}{\sigma_m \sqrt{2 \pi}}\right).

    Therefore, according to the similarity theorem, the equivalent :math:`F_1(u,v)` is

    .. math::

        F_1(u, v) = \sigma_l \sigma_m 2 \pi F_0 \left( \sigma_l \sqrt{2 \pi} u,\,
        \sigma_m \sqrt{2 \pi} v \right),

    or

    .. math::

        F_1(u, v) = a \sigma_l \sigma_m 2 \pi \exp \left ( -2 \pi^2 [\sigma_l^2 u^2 +
        \sigma_m^2 v^2] \right).

    Next, we rotate the Gaussian to match the sky plane rotation. A rotation
    :math:`\Omega` in the sky plane is carried out in the same direction in the
    Fourier plane,

    .. math::

        u' = u \cos \Omega - v \sin \Omega \\
        v' = u \sin \Omega + v \cos \Omega

    such that

    .. math::

        f_2(l, m) = f_1(l', m') \\
        F_2(u, v) = F_1(u', m')

    Finally, we translate the sky plane Gaussian by amounts :math:`\delta_l`,
    :math:`\delta_m`, which corresponds to a phase shift in the Fourier plane Gaussian.
    The image plane translation is

    .. math::

        f_3(l,m) = f_2(l - \delta_l, m - \delta_m)

    According to the shift theorem, the equivalent :math:`F_3(u,v)` is

    .. math::

        F_3(u,v) = \exp\left (- 2 i \pi [\delta_l u + \delta_m v] \right) F_2(u,v)

    We have arrived at the corresponding Fourier Gaussian, :math:`F_\mathrm{g}(u,v) =
    F_3(u,v)`. The simplified equation is

    .. math::

        F_\mathrm{g}(u,v) = a \sigma_l \sigma_m 2 \pi \exp \left (
        -2 \pi^2 \left [\sigma_l^2 u'^2 + \sigma_m^2 v'^2 \right]
        - 2 i \pi \left [\delta_l u + \delta_m v \right] \right).

    N.B. that we have mixed primed (:math:`u'`) and unprimed (:math:`u`) coordinates in
    the same equation for brevity.

    Finally, the same Fourier dual relationship holds

    .. math::

        f_\mathrm{g}(l,m) \leftrightharpoons F_\mathrm{g}(u,v)


    """

    # calculate primed rotated coordinates
    up = u * np.cos(Omega * deg) - v * np.sin(Omega * deg)
    vp = u * np.sin(Omega * deg) + v * np.cos(Omega * deg)

    # calculate the Fourier Gaussian
    fgauss: npt.NDArray[np.floating[Any]] = (
        a
        * sigma_l
        * sigma_m
        * 2
        * np.pi
        * np.exp(
            -2 * np.pi**2 * (sigma_l**2 * up**2 + sigma_m**2 * vp**2)
            - 2.0j * np.pi * (delta_l * u + delta_m * v)
        )
    )
    return fgauss


def fourier_gaussian_lambda_arcsec(
    u: npt.NDArray[np.floating[Any]],
    v: npt.NDArray[np.floating[Any]],
    a: float,
    delta_x: float,
    delta_y: float,
    sigma_x: float,
    sigma_y: float,
    Omega: float,
) -> npt.NDArray[np.floating[Any]]:
    r"""
    Calculate the Fourier plane Gaussian :math:`F_\mathrm{g}(u,v)` corresponding to the
    Sky plane Gaussian :math:`f_\mathrm{g}(l,m)` in
    :func:`~mpol.utils.sky_gaussian_arcsec`, using analytical relationships. The Fourier
    Gaussian is parameterized using the sky plane centroid (``delta_l, delta_m``),
    widths (``sigma_l, sigma_m``) and rotation (``Omega``). Assumes that ``a`` was in
    units of :math:`\mathrm{Jy}/\mathrm{arcsec}^2`.

    Args:
        u: l in units of [lambda]
        v: m in units of [lambda]
        a : amplitude prefactor, units of :math:`\mathrm{Jy}/\mathrm{arcsec}^2`.
        delta_x : offset [arcsec]
        delta_y : offset [arcsec]
        sigma_x : width [arcsec]
        sigma_y : width [arcsec]
        Omega : position angle of ascending node [degrees] east of north.

    Returns:
        2D Fourier Gaussian evaluated at input args
    """

    # convert the parameters and feed to the core routine
    return fourier_gaussian_lambda_radians(
        u,
        v,
        a / arcsec**2,
        delta_x * arcsec,
        delta_y * arcsec,
        sigma_x * arcsec,
        sigma_y * arcsec,
        Omega,
    )
