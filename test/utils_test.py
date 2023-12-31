import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpol import coordinates, utils
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
    coords = coordinates.GridCoords(cell_size=0.005, npix=800)

    ikw = {"origin": "lower"}

    g = utils.sky_gaussian_arcsec(
        coords.sky_x_centers_2D, coords.sky_y_centers_2D, **imagekw
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = ax.imshow(g, **ikw, extent=coords.img_ext)
    plt.colorbar(im, ax=ax)
    fig.savefig(tmp_path / "sky_gauss_2D.png", dpi=300)


def test_packed_gaussian(imagekw, tmp_path):
    coords = coordinates.GridCoords(cell_size=0.005, npix=800)

    ikw = {"origin": "lower"}

    g = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **imagekw
    )

    fig, ax = plt.subplots(nrows=1, ncols=1)
    im = ax.imshow(g, **ikw)
    plt.colorbar(im, ax=ax)
    fig.savefig(tmp_path / "packed_gauss_2D.png", dpi=300)


def test_analytic_plot(tmp_path):
    # plot the analytic Gaussian and its Fourier transform

    kw = {
        "a": 1,
        "delta_x": 0.02,  # arcsec
        "delta_y": -0.01,
        "sigma_x": 0.02,
        "sigma_y": 0.01,
        "Omega": 20,  # degrees
    }

    coords = coordinates.GridCoords(cell_size=0.005, npix=800)

    img = utils.sky_gaussian_arcsec(
        coords.sky_x_centers_2D, coords.sky_y_centers_2D, **kw
    )  # Jy/arcsec^2

    fig, ax = plt.subplots(nrows=1)
    im = ax.imshow(img, origin="lower")
    ax.set_xlabel("axis2 index")
    ax.set_ylabel("axis1 index")
    plt.colorbar(im, ax=ax, label=r"$Jy/\mathrm{arcsec}^2$")
    fig.savefig(tmp_path / "gaussian_sky.png", dpi=300)

    img_packed = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **kw
    )  # Jy/arcsec^2

    fig, ax = plt.subplots(nrows=1)
    ax.imshow(img_packed, origin="lower")
    ax.set_xlabel("axis2 index")
    ax.set_ylabel("axis1 index")
    fig.savefig(tmp_path / "gaussian_packed.png", dpi=300)

    # calculated the packed FFT
    fourier_packed_num = coords.cell_size**2 * np.fft.fft2(img_packed)

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

    diff_real = fourier_packed_an.real - fourier_packed_num.real
    diff_imag = fourier_packed_an.imag - fourier_packed_num.imag
    im = ax[2, 0].imshow(diff_real, **ikw)
    ax[2, 0].set_ylabel("difference")
    plt.colorbar(im, ax=ax[2, 0])
    im = ax[2, 1].imshow(diff_imag, **ikw)
    plt.colorbar(im, ax=ax[2, 1])

    fig.savefig(tmp_path / "fourier_packed.png", dpi=300)

    assert np.all(np.abs(diff_real) < 1e-12)
    assert np.all(np.abs(diff_imag) < 1e-12)


def test_loglinspace():
    # test that our log linspace routine calculates the correct spacing
    array = utils.loglinspace(0, 10, 5, 3)
    print(array)
    print(np.diff(array))
    assert len(array) == 5 + 3


def test_get_optimal_image_properties(baselines_1D):
    # test that get_optimal_image_properties returns sensible cell_size, npix
    image_width = 5.0 # [arcsec]

    u, v = baselines_1D

    cell_size, npix = utils.get_optimal_image_properties(image_width, u, v)

    max_data_freq = max(abs(u).max(), abs(v).max())

    assert(utils.get_max_spatial_freq(cell_size, npix) >= max_data_freq)