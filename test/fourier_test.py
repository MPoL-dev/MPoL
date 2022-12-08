import matplotlib.pyplot as plt
import torch

from mpol import fourier, utils
from mpol.constants import *


def test_fourier_layer(coords, tmp_path):
    # test image packing
    # test whether we get the same Fourier Transform using the FFT as we could
    # calculate analytically

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
    flayer = fourier.FourierCube(coords=coords)
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

    fig.savefig(tmp_path / "fourier_packed.png", dpi=300)

    assert np.all(np.abs(diff_real) < 1e-12)
    assert np.all(np.abs(diff_imag) < 1e-12)
    plt.close("all")


def test_fourier_grad(coords):
    # Test that we can calculate a gradient on a loss function using the Fourier layer

    kw = {
        "a": 1,
        "delta_x": 0.02,  # arcsec
        "delta_y": -0.01,
        "sigma_x": 0.02,
        "sigma_y": 0.01,
        "Omega": 20,  # degrees
    }

    img_packed = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **kw
    )

    # calculated the packed FFT using the FourierLayer
    flayer = fourier.FourierCube(coords=coords)
    # convert img_packed to pytorch tensor
    img_packed_tensor = torch.tensor(img_packed[np.newaxis, :, :], requires_grad=True)

    # calculated the packed FFT using the FourierLayer
    flayer = fourier.FourierCube(coords=coords)

    output = flayer.forward(img_packed_tensor)
    loss = torch.sum(torch.abs(output))

    loss.backward()
