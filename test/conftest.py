import pytest
import numpy as np
import requests
import os
from mpol import gridding

# We need a fixture which provides mock visibilities of the sort we'd
# expect from visread, but *without* the CASA dependency.

# fixture to provide tuple of uu, vv, weight, data_re, and data_im values
@pytest.fixture(scope="session")
def mock_visibility_data(tmp_path_factory):

    # check to see if the CACHE is defined AND we already have an NPZ file in the directory
    cache_path = os.getenv("MPOL_CACHE_DIR")
    if cache_path:
        npz_path = cache_path + "/logo_cube.npz"
        if os.path.exists(npz_path):
            print("using cached npz")
        else:
            # make the directory if it doesn't already exist
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)

            print("downloading npz to cache directory" + cache_path)
            # download the file here
            url = "https://zenodo.org/record/4498439/files/logo_cube.npz"
            r = requests.get(url)
            with open(npz_path, "wb") as f:
                f.write(r.content)
    else:
        # download the file to a temporary directory
        print(
            "Couldn't find MPOL_CACHE_DIR environment file, downloading npz to temporary directory"
        )
        npz_path = tmp_path_factory.mktemp("npz", numbered=False) / "logo_cube.npz"

        # download the file here
        url = "https://zenodo.org/record/4498439/files/logo_cube.npz"
        r = requests.get(url)
        with open(npz_path, "wb") as f:
            f.write(r.content)

    return np.load(npz_path)


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
    return gridding.GridCoords(cell_size=0.005, npix=800)


@pytest.fixture
def dataset(mock_visibility_data, coords):
    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data_re = d["data_re"]
    data_im = -d["data_im"]  # CASA convention

    gridder = gridding.Gridder(
        coords=coords, uu=uu, vv=vv, weight=weight, data_re=data_re, data_im=data_im,
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
        coords=coords, uu=uu, vv=vv, weight=weight, data_re=data_re, data_im=data_im,
    )
    gridder.grid_visibilities(weighting="uniform")

    return gridder.to_pytorch_dataset()
