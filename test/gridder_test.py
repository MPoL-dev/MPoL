import pytest
from mpol import gridding
import numpy as np


def test_grid_coords_instantiate():
    mycoords = gridding.GridCoords(cell_size=0.01, npix=512)


def test_grid_coords_odd_fail():
    with pytest.raises(AssertionError):
        mycoords = gridding.GridCoords(cell_size=0.01, npix=511)


def test_grid_coords_neg_cell_size():
    with pytest.raises(AssertionError):
        mycoords = gridding.GridCoords(cell_size=-0.01, npix=512)


# instantiate a Gridder object with mock visibilities
def test_grid_coords_fit(mock_visibility_data):
    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]

    mycoords = gridding.GridCoords(cell_size=0.005, npix=800)
    mycoords.check_data_fit(uu, vv)


def test_grid_coords_fail(mock_visibility_data):
    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]

    mycoords = gridding.GridCoords(cell_size=0.05, npix=800)

    print("max u data", np.max(uu))
    print("max u grid", mycoords.max_grid)

    with pytest.raises(AssertionError):
        mycoords.check_data_fit(uu, vv)


# if it works, create a fixture with the initialized visibilities for latter gridding ops
# @pytest.fixture
# def gridder(mock_visibility_data):
#     d = mock_visibility_data

#     uu = d["uu"]
#     vv = d["vv"]
#     weight = d["weight"]
#     data_re = d["data_re"]
#     data_im = d["data_im"]

#     cell_size = 0.005
#     npix = 800

#     return gridding.Gridder(cell_size, npix, uu, vv, weight, data_re, data_im)


# # actually do the gridding
# def test_grid_uniform(gridder):
#     gridder.grid_visibilities()


# def test_grid_natural(gridder):
#     gridder.grid_visibilities(weighting="natural")


# def test_grid_briggs(gridder):
#     gridder.grid_visibilities(weighting="briggs", robust=0.5)


# make a dirty beam

# make a dirty image

# see if we can get it in the correct units
# export them
# does the imager work for single-channeled visiblities?

