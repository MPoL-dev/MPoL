import pytest
import numpy as np
import matplotlib.pyplot as plt

from mpol import gridding
from mpol import utils
from mpol.constants import *


@pytest.fixture
def imagekw():
    return {
        "a": 1,
        "delta_x": 0.3,
        "delta_y": 0.1,
        "sigma_x": 0.3,
        "sigma_y": 0.1,
        "Omega": 20,
    }


def test_sky_gaussian(imagekw, tmp_path):
    coords = gridding.GridCoords(cell_size=0.005, npix=800)

    ikw = {"origin": "lower"}

    g = utils.sky_gaussian_arcsec(
        coords.sky_x_centers_2D, coords.sky_y_centers_2D, **imagekw
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = ax.imshow(g, **ikw, extent=coords.img_ext)
    plt.colorbar(im, ax=ax)
    fig.savefig(str(tmp_path / "sky_gauss_2D.png"), dpi=300)


def test_packed_gaussian(imagekw, tmp_path):
    coords = gridding.GridCoords(cell_size=0.005, npix=800)

    ikw = {"origin": "lower"}

    g = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **imagekw
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = ax.imshow(g, **ikw)
    plt.colorbar(im, ax=ax)
    fig.savefig(str(tmp_path / "packed_gauss_2D.png"), dpi=300)


def test_simple_gauss(tmp_path):
    coords = gridding.GridCoords(cell_size=100, npix=128)
    ikw = {"origin": "lower", "interpolation": "none"}

    l2D = coords.packed_x_centers_2D * arcsec  # [radians]
    m2D = coords.packed_y_centers_2D * arcsec  # [radians]

    u2D = coords.packed_u_centers_2D * 1e3  # [lambda]
    v2D = coords.packed_v_centers_2D * 1e3  # [lambda]

    img = np.exp(-np.pi * (l2D ** 2 + m2D ** 2))  # Jy/arcsec^2
    four_an = np.exp(-np.pi * (u2D ** 2 + v2D ** 2))  # Jy
    four_num = np.fft.fft2(img)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(6, 8))
    im = ax[0, 0].imshow(four_an.real, **ikw)
    plt.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title("real")
    ax[0, 0].set_ylabel("analytical")
    im = ax[0, 1].imshow(four_an.imag, **ikw)
    plt.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_title("imag")

    im = ax[1, 0].imshow(four_num.real, **ikw)
    plt.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_ylabel("numerical")
    im = ax[1, 1].imshow(four_num.imag, **ikw)
    plt.colorbar(im, ax=ax[1, 1])

    im = ax[2, 0].imshow(four_an.real - four_num.real, **ikw)
    ax[2, 0].set_ylabel("difference")
    plt.colorbar(im, ax=ax[2, 0])
    im = ax[2, 1].imshow(four_an.imag - four_num.imag, **ikw)
    plt.colorbar(im, ax=ax[2, 1])

    fig.savefig(str(tmp_path / "fourier_packed.png"), dpi=600)


def test_analytic_plot(tmp_path):
    kw = {
        "a": 1,
        "delta_x": 0.00,  # arcsec
        "delta_y": 0.00,
        "sigma_x": 1.0,
        "sigma_y": 1.0,
        "Omega": 0,  # degrees
    }

    coords = gridding.GridCoords(cell_size=0.05, npix=512)

    img = utils.sky_gaussian_arcsec(
        coords.sky_x_centers_2D, coords.sky_y_centers_2D, **kw
    )  # Jy/arcsec^2

    fig, ax = plt.subplots(nrows=1)
    ax.imshow(img, origin="lower")
    ax.set_xlabel("axis2 index")
    ax.set_ylabel("axis1 index")
    fig.savefig(str(tmp_path / "gaussian_sky.png"), dpi=300)

    img_packed = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **kw
    )  # Jy/arcsec^2

    fig, ax = plt.subplots(nrows=1)
    ax.imshow(img_packed, origin="lower")
    ax.set_xlabel("axis2 index")
    ax.set_ylabel("axis1 index")
    fig.savefig(str(tmp_path / "gaussian_packed.png"), dpi=300)

    # calculated the packed FFT
    fourier_packed_num = coords.cell_size ** 2 * np.fft.fft2(img_packed)

    # calculate the analytical FFT
    fourier_packed_an = utils.fourier_gaussian_klambda_arcsec(
        coords.packed_u_centers_2D, coords.packed_v_centers_2D, **kw
    )

    ikw = {"origin": "lower", "interpolation": "none"}

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(6, 8))
    im = ax[0, 0].imshow(fourier_packed_an.real, **ikw)
    plt.colorbar(im, ax=ax[0, 0])
    ax[0, 0].set_title("real")
    ax[0, 0].set_ylabel("analytical")
    im = ax[0, 1].imshow(fourier_packed_an.imag, **ikw)
    plt.colorbar(im, ax=ax[0, 1])
    ax[0, 1].set_title("imag")

    im = ax[1, 0].imshow(fourier_packed_num.real, **ikw)
    plt.colorbar(im, ax=ax[1, 0])
    ax[1, 0].set_ylabel("numerical")
    im = ax[1, 1].imshow(fourier_packed_num.imag, **ikw)
    plt.colorbar(im, ax=ax[1, 1])

    im = ax[2, 0].imshow(fourier_packed_an.real - fourier_packed_num.real, **ikw)
    ax[2, 0].set_ylabel("difference")
    plt.colorbar(im, ax=ax[2, 0])
    im = ax[2, 1].imshow(fourier_packed_an.imag - fourier_packed_num.imag, **ikw)
    plt.colorbar(im, ax=ax[2, 1])

    fig.savefig(str(tmp_path / "fourier_packed.png"), dpi=300)
