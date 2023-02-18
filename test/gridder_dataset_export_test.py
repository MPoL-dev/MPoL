import numpy as np
import pytest

from mpol import coordinates, gridding
from mpol.constants import *


# cache an instantiated DataAverager for future imaging ops
@pytest.fixture
def dataavg(mock_visibility_data):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    return gridding.DataAverager.from_image_properties(
        cell_size=0.005,
        npix=800,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )


def test_pytorch_export(dataavg):
    """
    Test that the dataset export routine doesn't error.
    """
    dataavg.to_pytorch_dataset()


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

    dataavg = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    with pytest.raises(RuntimeError):
        dataavg.to_pytorch_dataset()
