import matplotlib.pyplot as plt
import pytest
import torch
from astropy.io import fits

from mpol import gridding, images, utils
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


def test_basecube_grad():
    bcube = images.BaseCube(npix=800, cell_size=0.015)
    loss = torch.sum(bcube.forward())
    loss.backward()


def test_imagecube_grad(coords):

    bcube = images.BaseCube(coords=coords)
    # try passing through ImageLayer
    imagecube = images.ImageCube(coords=coords, passthrough=True)

    # send things through this layer
    loss = torch.sum(imagecube.forward(bcube.forward()))

    loss.backward()


# test for proper fits scale
def test_imagecube_tofits(coords, tmp_path):
    # creating base cube
    bcube = images.BaseCube(coords=coords)

    # try passing through ImageLayer
    imagecube = images.ImageCube(coords=coords, passthrough=True)

    # sending the basecube through the imagecube
    imagecube.forward(bcube.forward())

    # creating output fits file with name 'test_cube_fits_file39.fits'
    # file will be deleted after testing
    imagecube.to_FITS(fname=tmp_path / "test_cube_fits_file39.fits", overwrite=True)

    # inputting the header from the previously created fits file
    fits_header = fits.open(tmp_path / "test_cube_fits_file39.fits")[0].header
    assert (fits_header["CDELT1"] and fits_header["CDELT2"]) == pytest.approx(
        coords.cell_size / 3600
    )


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
    flayer = images.FourierCube(coords=coords)
    # convert img_packed to pytorch tensor
    img_packed_tensor = torch.tensor(img_packed[np.newaxis, :, :], requires_grad=True)

    # calculated the packed FFT using the FourierLayer
    flayer = images.FourierCube(coords=coords)

    output = flayer.forward(img_packed_tensor)
    loss = torch.sum(torch.abs(output))

    loss.backward()


def test_basecube_imagecube(coords, tmp_path):

    # create a mock cube that includes negative values
    nchan = 1
    mean = torch.full(
        (nchan, coords.npix, coords.npix), fill_value=-0.5, dtype=torch.double
    )
    std = torch.full(
        (nchan, coords.npix, coords.npix), fill_value=0.5, dtype=torch.double
    )

    # tensor
    base_cube = torch.normal(mean=mean, std=std)

    # layer
    basecube = images.BaseCube(coords=coords, nchan=nchan, base_cube=base_cube)

    # the default softplus function should map everything to positive values
    output = basecube.forward()

    fig, ax = plt.subplots(ncols=2, nrows=1)

    im = ax[0].imshow(
        np.squeeze(base_cube.detach().numpy()), origin="lower", interpolation="none"
    )
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title("input")

    im = ax[1].imshow(
        np.squeeze(output.detach().numpy()), origin="lower", interpolation="none"
    )
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title("mapped")

    fig.savefig(tmp_path / "basecube_mapped.png", dpi=300)

    # try passing through ImageLayer
    imagecube = images.ImageCube(coords=coords, nchan=nchan, passthrough=True)

    # send things through this layer
    imagecube.forward(basecube.forward())

    fig, ax = plt.subplots(ncols=1)
    im = ax.imshow(
        np.squeeze(imagecube.sky_cube.detach().numpy()),
        extent=imagecube.coords.img_ext,
        origin="lower",
        interpolation="none",
    )
    fig.savefig(tmp_path / "imagecube.png", dpi=300)

    plt.close("all")


def test_base_cube_conv_cube(coords, tmp_path):
    # test whether the HannConvCube functions appropriately

    # create a mock cube that includes negative values
    nchan = 1
    mean = torch.full(
        (nchan, coords.npix, coords.npix), fill_value=-0.5, dtype=torch.double
    )
    std = torch.full(
        (nchan, coords.npix, coords.npix), fill_value=0.5, dtype=torch.double
    )

    # The HannConvCube expects to function on a pre-packed ImageCube,
    # so in order to get the plots looking correct on this test image,
    # we need to faff around with packing

    # tensor
    test_cube = torch.normal(mean=mean, std=std)
    test_cube_packed = utils.sky_cube_to_packed_cube(test_cube)

    # layer
    conv_layer = images.HannConvCube(nchan=nchan)

    conv_output_packed = conv_layer(test_cube_packed)
    conv_output = utils.packed_cube_to_sky_cube(conv_output_packed)

    fig, ax = plt.subplots(ncols=2, nrows=1)

    im = ax[0].imshow(
        np.squeeze(test_cube.detach().numpy()), origin="lower", interpolation="none"
    )
    plt.colorbar(im, ax=ax[0])
    ax[0].set_title("input")

    im = ax[1].imshow(
        np.squeeze(conv_output.detach().numpy()), origin="lower", interpolation="none"
    )
    plt.colorbar(im, ax=ax[1])
    ax[1].set_title("convolved")

    fig.savefig(tmp_path / "convcube.png", dpi=300)

    plt.close("all")


def test_multi_chan_conv(coords, tmp_path):
    # create a mock channel cube that includes negative values
    # and make sure that the HannConvCube works across channels

    nchan = 10
    mean = torch.full(
        (nchan, coords.npix, coords.npix), fill_value=-0.5, dtype=torch.double
    )
    std = torch.full(
        (nchan, coords.npix, coords.npix), fill_value=0.5, dtype=torch.double
    )

    # tensor
    test_cube = torch.normal(mean=mean, std=std)

    # layer
    conv_layer = images.HannConvCube(nchan=nchan)

    conv_layer(test_cube)
