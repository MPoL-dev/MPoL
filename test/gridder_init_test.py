import pytest

from mpol import coordinates, gridding
from mpol.constants import *


def test_gridder_instantiate_cell_npix(mock_visibility_data):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    gridding.Gridder(
        cell_size=0.005,
        npix=800,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )


def test_gridder_instantiate_gridCoord(mock_visibility_data):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    mycoords = coordinates.GridCoords(cell_size=0.005, npix=800)

    gridding.Gridder(
        coords=mycoords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )


def test_gridder_instantiate_npix_gridCoord_conflict(mock_visibility_data):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    mycoords = coordinates.GridCoords(cell_size=0.005, npix=800)

    with pytest.raises(AssertionError):
        gridding.Gridder(
            cell_size=0.005,
            npix=800,
            coords=mycoords,
            uu=uu,
            vv=vv,
            weight=weight,
            data_re=data_re,
            data_im=data_im,
        )


def test_gridder_instantiate_bounds_fail(mock_visibility_data):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    mycoords = coordinates.GridCoords(cell_size=0.05, npix=800)

    with pytest.raises(AssertionError):
        gridding.Gridder(
            coords=mycoords,
            uu=uu,
            vv=vv,
            weight=weight,
            data_re=data_re,
            data_im=data_im,
        )
