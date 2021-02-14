import pytest
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpol import coordinates, connectors, datasets, gridding, images
from mpol.constants import *


# test instantiate connector
def test_instantiate_connector(coords, mock_visibility_data):
    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data_re = d["data_re"]
    data_im = -d["data_im"]

    gridder = gridding.Gridder(
        coords=coords, uu=uu, vv=vv, weight=weight, data_re=data_re, data_im=data_im,
    )
    gridder.grid_visibilities(weighting="uniform")

    dataset = gridder.to_pytorch_dataset()

    flayer = images.FourierCube(coords=coords)

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

    # try passing through ImageLayer
    imagecube = images.ImageCube(coords=coords, nchan=nchan, passthrough=True)

    dcon = connectors.DatasetConnector(flayer, dataset)

    # produce model visibilities
    vis = flayer.forward(imagecube.forward(basecube.forward()))

    # take a basecube, imagecube, and dataset and predict
    samples = dcon.forward(vis, dataset)


# see if we can backprop gradients
