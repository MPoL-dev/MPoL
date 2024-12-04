import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from astropy.io import fits
from mpol import coordinates, images, plot, utils
from plot_utils import imshow_two


def test_BaseCube_map(coords, tmp_path):
    # create a mock cube that includes negative values
    nchan = 1
    mean = torch.full((nchan, coords.npix, coords.npix), fill_value=-0.5)
    std = torch.full((nchan, coords.npix, coords.npix), fill_value=0.5)

    bcube = torch.normal(mean=mean, std=std)
    blayer = images.BaseCube(coords=coords, nchan=nchan, base_cube=bcube)

    # the default softplus function should map everything to positive values
    blayer_output = blayer()

    imshow_two(
        tmp_path / "BaseCube_mapped.png",
        [bcube, blayer_output],
        title=["BaseCube input", "BaseCube output"],
        xlabel=["pixel"],
        ylabel=["pixel"],
    )

    assert torch.all(blayer_output >= 0)


def test_instantiate_ImageCube():
    coords = coordinates.GridCoords(cell_size=0.015, npix=800)
    im = images.ImageCube(coords=coords)
    assert im.nchan == 1


def test_ImageCube_apply_grad(coords):
    bcube = images.BaseCube(coords=coords)
    imagecube = images.ImageCube(coords=coords)
    loss = torch.sum(imagecube(bcube()))
    loss.backward()


def test_to_FITS_pixel_scale(coords, tmp_path):
    """Test whether the FITS scale was written correctly."""
    bcube = images.BaseCube(coords=coords)
    imagecube = images.ImageCube(coords=coords)
    imagecube(bcube())

    # write FITS to file
    imagecube.to_FITS(fname=tmp_path / "test_cube_fits_file39.fits", overwrite=True)

    # read file and check pixel scale is correct
    fits_header = fits.open(tmp_path / "test_cube_fits_file39.fits")[0].header
    assert (fits_header["CDELT1"] and fits_header["CDELT2"]) == pytest.approx(
        coords.cell_size / 3600
    )


def test_HannConvCube(coords, tmp_path):
    # create a mock cube that includes negative values
    nchan = 1
    mean = torch.full((nchan, coords.npix, coords.npix), fill_value=-0.5)
    std = torch.full((nchan, coords.npix, coords.npix), fill_value=0.5)

    # The HannConvCube expects to function on a pre-packed ImageCube,
    test_cube = torch.normal(mean=mean, std=std)
    test_cube_packed = utils.sky_cube_to_packed_cube(test_cube)

    conv_layer = images.HannConvCube(nchan=nchan)

    conv_output_packed = conv_layer(test_cube_packed)
    conv_output = utils.packed_cube_to_sky_cube(conv_output_packed)

    imshow_two(
        tmp_path / "convcube.png",
        [test_cube, conv_output],
        title=["input", "convolved"],
        xlabel=["pixel"],
        ylabel=["pixel"],
    )


def test_HannConvCube_multi_chan(coords):
    """Make sure HannConvCube functions with multi-channeled input"""
    nchan = 10
    mean = torch.full((nchan, coords.npix, coords.npix), fill_value=-0.5)
    std = torch.full((nchan, coords.npix, coords.npix), fill_value=0.5)

    test_cube = torch.normal(mean=mean, std=std)

    conv_layer = images.HannConvCube(nchan=nchan)
    conv_layer(test_cube)


def test_flux(coords):
    """Make sure we can read the flux attribute."""
    nchan = 20
    bcube = images.BaseCube(coords=coords, nchan=nchan)
    im = images.ImageCube(coords=coords, nchan=nchan)
    im(bcube())
    assert im.flux.size()[0] == nchan


def test_plot_test_img(packed_cube, coords, tmp_path):
    # show only the first channel
    chan = 0
    fig, ax = plt.subplots(nrows=1)

    # put back to sky
    sky_cube = utils.packed_cube_to_sky_cube(packed_cube)
    im = ax.imshow(sky_cube[chan], extent=coords.img_ext, origin="lower")
    plt.colorbar(im)
    fig.savefig(tmp_path / "sky_cube.png")

    plt.close("all")


def test_uv_gaussian_taper(coords, tmp_path):
    for r in np.arange(0.0, 0.2, step=0.04):
        fig, ax = plt.subplots(ncols=1)

        taper_2D = images.uv_gaussian_taper(coords, r, r, 0.0)
        print(type(taper_2D))

        norm = plot.get_image_cmap_norm(taper_2D, symmetric=True)
        im = ax.imshow(
            taper_2D,
            extent=coords.vis_ext_Mlam,
            origin="lower",
            cmap="bwr_r",
            norm=norm,
        )
        plt.colorbar(im, ax=ax)

        fig.savefig(tmp_path / f"taper{r:.2f}.png")

    plt.close("all")


def test_GaussConvImage_kernel(coords, tmp_path):
    rs = np.array([0.02, 0.06, 0.10])
    nchan = 3
    fig, ax = plt.subplots(nrows=len(rs), ncols=nchan, figsize=(10, 10))
    for i, r in enumerate(rs):
        layer = images.GaussConvImage(coords, nchan=nchan, FWHM_maj=r, FWHM_min=0.5 * r)
        weight = layer.m.weight.detach().numpy()
        for j in range(nchan):
            im = ax[i, j].imshow(weight[j, 0], interpolation="none", origin="lower")
            plt.colorbar(im, ax=ax[i, j])

    fig.savefig(tmp_path / "filter.png")
    plt.close("all")


def test_GaussConvImage_kernel_rotate(coords, tmp_path):
    r = 0.04
    Omegas = [0, 20, 40]  # degrees
    nchan = 3
    fig, ax = plt.subplots(nrows=len(Omegas), ncols=nchan, figsize=(10, 10))
    for i, Omega in enumerate(Omegas):
        layer = images.GaussConvImage(
            coords, nchan=nchan, FWHM_maj=r, FWHM_min=0.5 * r, Omega=Omega
        )
        weight = layer.m.weight.detach().numpy()
        for j in range(nchan):
            im = ax[i, j].imshow(weight[j, 0], interpolation="none", origin="lower")
            plt.colorbar(im, ax=ax[i, j])

    fig.savefig(tmp_path / "filter.png")
    plt.close("all")


@pytest.mark.parametrize("FWHM", [0.02, 0.06, 0.1])
def test_GaussConvImage(sky_cube, coords, tmp_path, FWHM):
    chan = 0
    nchan = sky_cube.size()[0]

    layer = images.GaussConvImage(coords, nchan=nchan, FWHM_maj=FWHM, FWHM_min=FWHM)
    c_sky = layer(sky_cube)

    imgs = [sky_cube[chan], c_sky[chan]]
    fluxes = [coords.cell_size**2 * torch.sum(img).item() for img in imgs]
    title = [f"tot flux: {flux:.3f} Jy" for flux in fluxes]

    imshow_two(
        tmp_path / f"convolved_{FWHM:.2f}.png",
        imgs,
        sky=True,
        suptitle=f"Image Plane Gauss Convolution FWHM={FWHM}",
        title=title,
        extent=[coords.img_ext],
    )

    assert pytest.approx(fluxes[0]) == fluxes[1]


@pytest.mark.parametrize("Omega", [0, 15, 30, 45])
def test_GaussConvImage_rotate(sky_cube, coords, tmp_path, Omega):
    chan = 0
    nchan = sky_cube.size()[0]

    FWHM_maj = 0.10
    FWHM_min = 0.05

    layer = images.GaussConvImage(
        coords, nchan=nchan, FWHM_maj=FWHM_maj, FWHM_min=FWHM_min, Omega=Omega
    )
    c_sky = layer(sky_cube)

    imgs = [sky_cube[chan], c_sky[chan]]
    fluxes = [coords.cell_size**2 * torch.sum(img).item() for img in imgs]
    title = [f"tot flux: {flux:.3f} Jy" for flux in fluxes]

    imshow_two(
        tmp_path / f"convolved_{Omega:.0f}_deg.png",
        imgs,
        sky=True,
        suptitle=r"Image Plane Gauss Convolution: $\Omega$="
        + f'{Omega}, {FWHM_maj}", {FWHM_min}"',
        title=title,
        extent=[coords.img_ext],
    )

    assert pytest.approx(fluxes[0], abs=4e-7) == fluxes[1]


@pytest.mark.parametrize("FWHM", [0.02, 0.1, 0.2, 0.3, 0.5])
def test_GaussConvFourier(packed_cube, coords, tmp_path, FWHM):
    chan = 0
    sky_cube = utils.packed_cube_to_sky_cube(packed_cube)

    layer = images.GaussConvFourier(coords, FWHM, FWHM)
    c = layer(packed_cube)
    c_sky = utils.packed_cube_to_sky_cube(c)

    imgs = [sky_cube[chan], c_sky[chan]]
    fluxes = [coords.cell_size**2 * torch.sum(img).item() for img in imgs]
    title = [f"tot flux: {flux:.3f} Jy" for flux in fluxes]

    imshow_two(
        tmp_path / "convolved_FWHM_{:.2f}.png".format(FWHM),
        imgs,
        sky=True,
        suptitle=f"Fourier Plane Gauss Convolution: FWHM={FWHM}",
        title=title,
        extent=[coords.img_ext],
    )

    assert pytest.approx(fluxes[0], abs=4e-7) == fluxes[1]


@pytest.mark.parametrize("Omega", [0, 15, 30, 45])
def test_GaussConvFourier_rotate(packed_cube, coords, tmp_path, Omega):
    chan = 0
    sky_cube = utils.packed_cube_to_sky_cube(packed_cube)

    FWHM_maj = 0.10
    FWHM_min = 0.05
    layer = images.GaussConvFourier(
        coords, FWHM_maj=FWHM_maj, FWHM_min=FWHM_min, Omega=Omega
    )

    c = layer(packed_cube)
    c_sky = utils.packed_cube_to_sky_cube(c)

    imgs = [sky_cube[chan], c_sky[chan]]
    fluxes = [coords.cell_size**2 * torch.sum(img).item() for img in imgs]
    title = [f"tot flux: {flux:.3f} Jy" for flux in fluxes]

    imshow_two(
        tmp_path / f"convolved_{Omega:.0f}_deg.png",
        imgs,
        sky=True,
        suptitle=r"Fourier Plane Gauss Convolution: $\Omega$="
        + f'{Omega}, {FWHM_maj}", {FWHM_min}"',
        title=title,
        extent=[coords.img_ext],
    )

    assert pytest.approx(fluxes[0], abs=4e-7) == fluxes[1]


def test_GaussConvFourier_point(coords, tmp_path):
    FWHM = 0.5

    # create an image with a point source in the center
    sky_cube = torch.zeros((1, coords.npix, coords.npix))
    cpix = coords.npix // 2
    sky_cube[0, cpix, cpix] = 1.0

    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True)
    # put back to sky
    im = ax[0].imshow(sky_cube[0], extent=coords.img_ext, origin="lower")
    flux = coords.cell_size**2 * torch.sum(sky_cube[0])
    ax[0].set_title(f"tot flux: {flux:.3f} Jy")
    plt.colorbar(im, ax=ax[0])

    # set base resolution
    layer = images.GaussConvFourier(coords, FWHM, FWHM)
    packed_cube = utils.sky_cube_to_packed_cube(sky_cube)
    c = layer(packed_cube)
    # put back to sky
    c_sky = utils.packed_cube_to_sky_cube(c)
    flux = coords.cell_size**2 * torch.sum(c_sky[0])
    im = ax[1].imshow(
        c_sky[0].detach().numpy(),
        extent=coords.img_ext,
        origin="lower",
        cmap="inferno",
    )
    ax[1].set_title(f"tot flux: {flux:.3f} Jy")
    r = 0.7
    ax[1].set_xlim(r, -r)
    ax[1].set_ylim(-r, r)

    plt.colorbar(im, ax=ax[1])
    fig.savefig(tmp_path / "point_source_FWHM_{:.2f}.png".format(FWHM))

    plt.close("all")
