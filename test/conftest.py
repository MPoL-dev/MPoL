import numpy as np
import pytest
from astropy.utils.data import download_file

from mpol import coordinates, gridding

# We need a fixture which provides mock visibilities of the sort we'd
# expect from visread, but *without* the CASA dependency.

# fixture to provide tuple of uu, vv, weight, data_re, and data_im values
@pytest.fixture(scope="session")
def mock_visibility_data():

    # use astropy routines to cache data
    fname = download_file(
        "https://zenodo.org/record/4498439/files/logo_cube.npz",
        cache=True,
        pkgname="mpol",
    )

    return np.load(fname)


@pytest.fixture
def mock_visibility_data_cont(mock_visibility_data):
    chan = 4
    d = mock_visibility_data
    uu = d["uu"][chan]
    vv = d["vv"][chan]
    weight = d["weight"][chan]
    data_re = d["data_re"][chan]
    data_im = -d["data_im"][chan]  # CASA convention

    return uu, vv, weight, data_re, data_im


@pytest.fixture
def coords():
    return coordinates.GridCoords(cell_size=0.005, npix=800)


@pytest.fixture
def dataset(mock_visibility_data, coords):
    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data_re = d["data_re"]
    data_im = -d["data_im"]  # CASA convention

    gridder = gridding.Gridder(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )
    gridder.grid_visibilities(weighting="uniform")

    return gridder.to_pytorch_dataset()


@pytest.fixture
def dataset_cont(mock_visibility_data, coords):

    chan = 4
    d = mock_visibility_data
    uu = d["uu"][chan]
    vv = d["vv"][chan]
    weight = d["weight"][chan]
    data_re = d["data_re"][chan]
    data_im = -d["data_im"][chan]  # CASA convention

    gridder = gridding.Gridder(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )
    gridder.grid_visibilities(weighting="uniform")

    return gridder.to_pytorch_dataset()


@pytest.fixture
def crossvalidation_products(mock_visibility_data):
    # test the crossvalidation with a smaller set of coordinates than normal,
    # better matched to the extremes of the mock dataset
    coords = coordinates.GridCoords(cell_size=0.04, npix=256)

    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data_re = d["data_re"]
    data_im = -d["data_im"]  # CASA convention

    gridder = gridding.Gridder(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )
    gridder.grid_visibilities(weighting="uniform")
    dataset = gridder.to_pytorch_dataset()

    return coords, dataset
