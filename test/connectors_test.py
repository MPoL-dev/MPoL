import matplotlib.pyplot as plt
import numpy as np
import torch

from mpol import datasets, fourier, images
from mpol.constants import *


def test_index_vis(coords, dataset):
    # test that we can index a dataset

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

    # take a basecube, imagecube, and GriddedDataset and predict corresponding visibilities.
    datasets.index_vis(vis, dataset)


def test_connector_grad(coords, dataset):
    # test that we can calculate the gradients through the loss

    flayer = fourier.FourierCube(coords=coords)
    nchan = dataset.nchan
    basecube = images.BaseCube(coords=coords, nchan=nchan)
    imagecube = images.ImageCube(coords=coords, nchan=nchan, passthrough=True)

    # produce model visibilities
    vis = flayer.forward(imagecube.forward(basecube.forward()))
    samples = datasets.index_vis(vis, dataset)

    print(samples)
    loss = torch.sum(torch.abs(samples))

    # segfaults on 3.9
    # https://github.com/pytorch/pytorch/issues/50014
    loss.backward()

    print(basecube.base_cube.grad)
