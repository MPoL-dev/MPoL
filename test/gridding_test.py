import pytest
import numpy as np
import matplotlib.pyplot as plt

from mpol import gridding
from mpol.constants import *


def sky_plane(
    alpha,
    dec,
    a=1,
    delta_alpha=1.0 * arcsec,
    delta_delta=1.0 * arcsec,
    sigma_alpha=1.0 * arcsec,
    sigma_delta=1.0 * arcsec,
    Omega=0.0,
):
    """
    Calculates a Gaussian on the sky plane 

    Args:
        alpha: ra (in radians)
        delta: dec (in radians)
        a : amplitude
        delta_alpha : offset (in radians)
        delta_dec : offset (in radians)
        sigma_alpha : width (in radians)
        sigma_dec : width (in radians)
        Omega : position angle of ascending node (in degrees east of north)

    Returns:
        Gaussian evaluated at input args
    """
    return a * np.exp(
        -(
            (alpha - delta_alpha) ** 2 / (2 * sigma_alpha ** 2)
            + (dec - delta_delta) ** 2 / (2 * sigma_delta ** 2)
        )
    )


def fourier_plane(
    u,
    v,
    a=1,
    delta_alpha=1.0 * arcsec,
    delta_delta=1.0 * arcsec,
    sigma_alpha=1.0 * arcsec,
    sigma_delta=1.0 * arcsec,
    Omega=0.0,
):
    """
    Calculates the Analytic Fourier transform of the sky plane Gaussian. 

    Args:
        u: spatial freq (in kλ)
        v: spatial freq (in  kλ)
        a : amplitude
        delta_alpha : offset (in radians)
        delta_dec : offset (in radians)
        sigma_alpha : width (in radians)
        sigma_dec : width (in radians)
        Omega : position angle of ascending node (in degrees east of north)
    
    Returns:
        FT Gaussian evaluated at u, v points
    """

    # convert back to radians
    u = u * 1e3
    v = v * 1e3

    return (
        2
        * np.pi
        * a
        * sigma_alpha
        * sigma_delta
        * np.exp(
            -2 * np.pi ** 2 * (sigma_alpha ** 2 * u ** 2 + sigma_delta ** 2 * v ** 2)
            - 2 * np.pi * 1.0j * (delta_alpha * u + delta_delta * v)
        )
    )


@pytest.fixture(scope="module")
def image_dict():

    N_alpha = 128
    N_dec = 128
    img_radius = 15.0 * arcsec

    # full span of the image
    ra = gridding.fftspace(img_radius, N_alpha)  # [arcsec]
    dec = gridding.fftspace(img_radius, N_dec)  # [arcsec]

    # fill out an image
    img = np.empty((N_dec, N_alpha), np.float)

    for i, delta in enumerate(dec):
        for j, alpha in enumerate(ra):
            img[i, j] = sky_plane(alpha, delta)

    return {
        "N_alpha": N_alpha,
        "N_dec": N_dec,
        "img_radius": img_radius,
        "ra": ra,
        "dec": dec,
        "img": img,
    }


@pytest.fixture(scope="module")
def corrfun_mat(image_dict):
    ra = image_dict["ra"]
    dec = image_dict["dec"]

    # pre-multiply the image by the correction function
    return gridding.corrfun_mat(np.fft.fftshift(ra), np.fft.fftshift(dec))


def test_plot_input(image_dict, tmp_path):
    fig, ax = plt.subplots(nrows=1)
    ax.imshow(image_dict["img"], origin="upper", interpolation="none", aspect="equal")
    ax.set_xlabel(r"$\Delta \alpha \cos \delta$")
    ax.set_ylabel(r"$\Delta \delta$")
    ax.set_title("Input image")
    fig.savefig(str(tmp_path / "input.png"), dpi=300)


def test_fill_corrfun_matrix(corrfun_mat, tmp_path):

    fig, ax = plt.subplots(nrows=1)
    im = ax.imshow(corrfun_mat, origin="upper", interpolation="none", aspect="equal")
    plt.colorbar(im)
    ax.set_xlabel(r"$\Delta \alpha \cos \delta$")
    ax.set_ylabel(r"$\Delta \delta$")
    ax.set_title("Correction function")
    fig.savefig(str(tmp_path / "corrfun.png"), dpi=300)


@pytest.fixture(scope="module")
def vis_dict(image_dict, corrfun_mat):

    img = image_dict["img"]

    # calculate the Fourier coordinates
    dalpha = (2 * image_dict["img_radius"]) / image_dict["N_alpha"]
    ddelta = (2 * image_dict["img_radius"]) / image_dict["N_dec"]

    us = np.fft.rfftfreq(image_dict["N_alpha"], d=dalpha) * 1e-3  # convert to [kλ]
    vs = np.fft.fftfreq(image_dict["N_dec"], d=ddelta) * 1e-3  # convert to [kλ]

    # calculate the FFT, but first shift all axes.
    # normalize output properly
    vis = (
        dalpha * ddelta * np.fft.rfftn(corrfun_mat * np.fft.fftshift(img), axes=(0, 1))
    )
    vis_no_cor = dalpha * ddelta * np.fft.rfftn(np.fft.fftshift(img), axes=(0, 1))

    return {"us": us, "vs": vs, "vis": vis, "vis_no_cor": vis_no_cor}


@pytest.fixture(scope="module")
def vis_analytical_full(vis_dict):
    vs = vis_dict["vs"]

    XX_full, YY_full = np.meshgrid(vs, vs)
    return fourier_plane(XX_full, YY_full)


@pytest.fixture(scope="module")
def vis_analytical_half(vis_dict):
    us = vis_dict["us"]
    vs = vis_dict["vs"]

    XX_full, YY_full = np.meshgrid(us, vs)
    return fourier_plane(XX_full, YY_full)


def test_plot_full_analytical(tmp_path, vis_dict, vis_analytical_full):
    vs_limit = np.fft.fftshift(vis_dict["vs"])

    ext_full = [vs_limit[-1], vs_limit[0], vs_limit[-1], vs_limit[0]]
    fig, ax = plt.subplots(nrows=2)
    ax[0].imshow(
        np.real(np.fft.fftshift(vis_analytical_full)), origin="upper", extent=ext_full
    )
    ax[1].imshow(
        np.imag(np.fft.fftshift(vis_analytical_full)), origin="upper", extent=ext_full
    )
    fig.savefig(str(tmp_path / "analytical_full.png"), dpi=300)


@pytest.fixture(scope="module")
def vis_diff(vis_dict, vis_analytical_half):
    # compare to the analytic version
    vis_no_cor = vis_dict["vis_no_cor"]

    return np.fft.fftshift(vis_no_cor - vis_analytical_half, axes=0)


def test_vis_diff_ok(vis_diff):
    # create a numerical test to make sure the difference is small
    assert np.sum(np.abs(vis_diff)) < 1e-10


def test_compare_analytical_numerical(
    tmp_path, vis_dict, vis_analytical_half, vis_diff
):
    vis_no_cor = vis_dict["vis_no_cor"]

    us = vis_dict["us"]
    vs_limit = np.fft.fftshift(vis_dict["vs"])
    ext = [us[0], us[-1], vs_limit[-1], vs_limit[0]]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(7, 5))
    ax[0, 0].set_title("numerical")
    ax[0, 0].imshow(
        np.real(np.fft.fftshift(vis_no_cor, axes=0)),
        origin="upper",
        interpolation="none",
        aspect="equal",
        extent=ext,
    )
    ax[1, 0].imshow(
        np.imag(np.fft.fftshift(vis_no_cor, axes=0)),
        origin="upper",
        interpolation="none",
        aspect="equal",
        extent=ext,
    )

    ax[0, 1].set_title("analytical")
    im_ral = ax[0, 1].imshow(
        np.real(np.fft.fftshift(vis_analytical_half, axes=0)),
        origin="upper",
        interpolation="none",
        aspect="equal",
        extent=ext,
    )
    plt.colorbar(im_ral, ax=ax[0, 1])
    ax[1, 1].imshow(
        np.imag(np.fft.fftshift(vis_analytical_half, axes=0)),
        origin="upper",
        interpolation="none",
        aspect="equal",
        extent=ext,
    )

    np.imag(np.fft.fftshift(vis_no_cor - vis_analytical_half, axes=0))
    ax[0, 2].set_title("difference")
    im_real = ax[0, 2].imshow(
        np.real(vis_diff),
        origin="upper",
        interpolation="none",
        aspect="equal",
        extent=ext,
    )
    plt.colorbar(im_real, ax=ax[0, 2])
    im_imag = ax[1, 2].imshow(
        np.imag(vis_diff),
        origin="upper",
        interpolation="none",
        aspect="equal",
        extent=ext,
    )
    plt.colorbar(im_imag, ax=ax[1, 2])
    fig.subplots_adjust(wspace=0.05)

    fig.savefig(str(tmp_path / "comparison.png"), dpi=300)


@pytest.fixture(scope="module")
def baselines():
    return np.array(
        [
            [50.0, 10.0],
            [50.0, 0.0],
            [50.0, -1.0],
            [-50.0, 10.0],
            [5.0, 1.0],
            [-5.0, 1.0],
            [5.0, 20.0],
            [-5.0, -20.0],
        ]
    )


@pytest.fixture(scope="module")
def analytic_samples(baselines):
    u_data = baselines[:, 0]
    v_data = baselines[:, 1]
    return fourier_plane(u_data, v_data)


@pytest.fixture(scope="module")
def interpolation_matrices(vis_dict, baselines):
    u_data = baselines[:, 0]
    v_data = baselines[:, 1]
    us = vis_dict["us"]
    vs = vis_dict["vs"]

    # calculate and visualize the C_real and C_imag matrices
    # these are scipy csc sparse matrices
    return gridding.calc_matrices(u_data, v_data, us, vs)


def test_plot_interpolation_matrices(tmp_path, interpolation_matrices):
    C_real, C_imag = interpolation_matrices

    C_real = C_real.toarray()
    C_imag = C_imag.toarray()

    fig, ax = plt.subplots(nrows=1, figsize=(6, 6))
    vvmax = np.max(np.abs(C_real[:, 0:300]))
    ax.imshow(
        C_real[:, 0:300],
        interpolation="none",
        origin="upper",
        cmap="RdBu",
        aspect="auto",
        vmin=-vvmax,
        vmax=vvmax,
    )
    # ax[1].spy(C_real[:, 0:300], marker=".", precision="present", aspect="auto")
    fig.savefig(str(tmp_path / "C_real.png"), dpi=300)

    fig, ax = plt.subplots(ncols=1, figsize=(6, 6))
    vvmax = np.max(np.abs(C_imag[:, 0:300]))
    ax.imshow(
        C_imag[:, 0:300],
        interpolation="none",
        origin="upper",
        cmap="RdBu",
        aspect="auto",
        vmin=-vvmax,
        vmax=vvmax,
    )
    fig.savefig(str(tmp_path / "C_imag.png"), dpi=300)


def test_interpolate_points(vis_dict, interpolation_matrices):
    vis = vis_dict["vis"]
    C_real, C_imag = interpolation_matrices
    # get interpolated points (TODO: check that they are correct...)
    C_real.dot(np.real(vis.flatten()))
    C_imag.dot(np.imag(vis.flatten()))


@pytest.fixture(scope="module")
def interpolated_points(vis_dict, interpolation_matrices):
    vis = vis_dict["vis"]
    C_real, C_imag = interpolation_matrices
    return C_real.dot(np.real(vis.flatten())), C_imag.dot(np.imag(vis.flatten()))


def test_plot_interpolate_points(tmp_path, analytic_samples, interpolated_points):
    interp_real, interp_imag = interpolated_points
    fig, ax = plt.subplots(nrows=4, figsize=(4, 5))
    ax[0].plot(np.real(analytic_samples), ".", ms=4)
    ax[0].plot(interp_real, ".", ms=3)
    ax[0].set_ylabel("real")
    ax[1].plot(interp_real - np.real(analytic_samples), ".")
    ax[1].set_ylabel("real diff")
    ax[2].plot(np.imag(analytic_samples), ".", ms=4)
    ax[2].plot(interp_imag, ".", ms=3)
    ax[2].set_ylabel("imag")
    ax[3].plot(interp_imag - np.imag(analytic_samples), ".")
    ax[3].set_ylabel("imag diff")
    fig.subplots_adjust(hspace=0.4, left=0.2)
    fig.savefig(str(tmp_path / "real_comp.png"), dpi=300)
