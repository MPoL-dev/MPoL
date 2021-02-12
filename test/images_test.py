import pytest
import torch
import matplotlib.pyplot as plt

from mpol import images
from mpol import gridding
from mpol import utils
from mpol.constants import *


def test_odd_npix():
    with pytest.raises(AssertionError):
        images.BaseCube(npix=853, nchan=30, cell_size=0.015)

    with pytest.raises(AssertionError):
        images.ImageCube(npix=853, nchan=30, cell_size=0.015)


def test_negative_cell_size():
    with pytest.raises(AssertionError):
        images.BaseCube(npix=800, nchan=30, cell_size=-0.015)

    with pytest.raises(AssertionError):
        images.ImageCube(npix=800, nchan=30, cell_size=-0.015)


def test_single_chan():
    im = images.ImageCube(cell_size=0.015, npix=800)
    assert im.nchan == 1


@pytest.fixture
def coords():
    return gridding.GridCoords(cell_size=0.005, npix=800)


# test image packing
def test_fourier_layer(coords, tmp_path):
    kw = {
        "a": 1,
        "delta_x": 0.02,  # arcsec
        "delta_y": -0.01,
        "sigma_x": 0.02,
        "sigma_y": 0.01,
        "Omega": 20,  # degrees
    }

    img = utils.sky_gaussian_arcsec(
        coords.sky_x_centers_2D, coords.sky_y_centers_2D, **kw
    )

    img_packed = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **kw
    )

    # calculated the packed FFT using the FourierLayer
    flayer = images.FourierCube(coords=coords)
    # convert img_packed to pytorch tensor
    img_packed_tensor = torch.from_numpy(img_packed[np.newaxis, :, :])
    fourier_packed_num = np.squeeze(flayer.forward(img_packed_tensor).numpy())

    # calculate the analytical FFT
    fourier_packed_an = utils.fourier_gaussian_klambda_arcsec(
        coords.packed_u_centers_2D, coords.packed_v_centers_2D, **kw
    )

    ikw = {"origin": "lower"}

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

    fig.savefig(str(tmp_path / "fourier_packed.png"), dpi=300)

    assert np.all(np.abs(diff_real) < 1e-12)
    assert np.all(np.abs(diff_imag) < 1e-12)


# test basecube pixel mapping
# using known input cube, known mapping function, compare output
