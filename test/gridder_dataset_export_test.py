import numpy as np
import pytest
from mpol import coordinates, gridding


def test_cell_variance_error_pytorch(mock_dataset_np):
    """
    Test that the gridder routine errors if we send it data that has the wrong scatter relative to the weight values.
    """
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_dataset_np
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=2 * sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(
        loc=0, scale=2 * sigma, size=uu.shape
    )

    averager = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    with pytest.raises(RuntimeError):
        averager.to_pytorch_dataset()
