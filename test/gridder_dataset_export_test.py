import numpy as np
import torch
import pytest

from mpol import coordinates, gridding
from mpol.constants import *


# cache an instantiated gridder for future imaging ops
@pytest.fixture
def gridder(mock_visibility_data):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    return gridding.Gridder(
        cell_size=0.005,
        npix=800,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )


def test_pytorch_export(gridder, device=None):
    """
    Test that the dataset export routine doesn't error.
    """
    gridder.to_pytorch_dataset(device)


def test_pytorch_export_gpu(gridder):
    """
    Test that the dataset export routine doesn't error when using the GPU.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    
        gridder.to_pytorch_dataset(gridder, device=device)

    else:
        pass


def test_cell_variance_error_pytorch(mock_visibility_data):
    """
    Test that the gridder routine errors if we send it data that has the wrong scatter relative to the weight values.
    """
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_visibility_data
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=2 * sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(
        loc=0, scale=2 * sigma, size=uu.shape
    )

    gridder = gridding.Gridder(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    with pytest.raises(RuntimeError):
        gridder.to_pytorch_dataset()
