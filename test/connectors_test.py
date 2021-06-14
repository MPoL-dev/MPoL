import matplotlib.pyplot as plt
import numpy as np
import torch

from mpol import connectors, coordinates, fourier, images
from mpol.constants import *


# test instantiate connector
def test_instantiate_connector(coords, dataset):

    flayer = fourier.FourierCube(coords=coords)

    # create a mock cube that includes negative values
    nchan = dataset.nchan
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

    # try passing through ImageLayer
    imagecube = images.ImageCube(coords=coords, nchan=nchan, passthrough=True)

    # produce model visibilities
    vis = flayer.forward(imagecube.forward(basecube.forward()))

    # take a basecube, imagecube, and dataset and predict
    connectors.index_vis(vis, dataset)


def test_connector_grad(coords, dataset):

    flayer = fourier.FourierCube(coords=coords)
    nchan = dataset.nchan
    basecube = images.BaseCube(coords=coords, nchan=nchan)
    imagecube = images.ImageCube(coords=coords, nchan=nchan, passthrough=True)

    # produce model visibilities
    vis = flayer.forward(imagecube.forward(basecube.forward()))
    samples = connectors.index_vis(vis, dataset)

    print(samples)
    loss = torch.sum(torch.abs(samples))

    # segfaults on 3.9
    # https://github.com/pytorch/pytorch/issues/50014
    loss.backward()

    print(basecube.base_cube.grad)


def test_residual_connector(coords, dataset_cont, tmp_path):

    flayer = fourier.FourierCube(coords=coords)

    # create a mock cube that includes negative values
    nchan = dataset_cont.nchan

    # tensor
    cube = torch.full(
        (nchan, coords.npix, coords.npix), fill_value=0.0, dtype=torch.double
    )

    # try passing through ImageLayer
    imagecube = images.ImageCube(coords=coords, nchan=nchan, cube=cube)

    # produce model visibilities to store vis to flayer
    flayer.forward(imagecube.forward())

    # instantiate residual connector
    rcon = connectors.GriddedResidualConnector(flayer, dataset_cont)

    # store residual products
    rcon.forward()

    # plot residual image compared to imagecube.image
    fig, ax = plt.subplots(ncols=2, nrows=2)
    im = ax[0, 0].imshow(
        np.squeeze(imagecube.sky_cube.detach().numpy()),
        origin="lower",
        interpolation="none",
        extent=imagecube.coords.img_ext,
    )
    ax[0, 0].set_title("ImageCube")
    plt.colorbar(im, ax=ax[0, 0])

    im = ax[0, 1].imshow(
        np.squeeze(rcon.sky_cube.detach().numpy()),
        origin="lower",
        interpolation="none",
        extent=rcon.coords.img_ext,
    )
    ax[0, 1].set_title("ResidualImage")
    plt.colorbar(im, ax=ax[0, 1])

    im = ax[1, 0].imshow(
        np.squeeze(rcon.ground_amp.detach().numpy()),
        origin="lower",
        interpolation="none",
        extent=imagecube.coords.vis_ext,
    )
    ax[1, 0].set_title("Amplitude")
    plt.colorbar(im, ax=ax[1, 0])

    im = ax[1, 1].imshow(
        np.squeeze(rcon.ground_phase.detach().numpy()),
        origin="lower",
        interpolation="none",
        extent=imagecube.coords.vis_ext,
    )
    ax[1, 1].set_title("Phase")
    plt.colorbar(im, ax=ax[1, 1])

    fig.subplots_adjust(left=0.1, right=0.9, wspace=0.3, hspace=0.3, top=0.9)
    fig.savefig(tmp_path / "residual.png", dpi=300)
    plt.close("all")
