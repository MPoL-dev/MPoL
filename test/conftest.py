import numpy as np
import pytest
from astropy.utils.data import download_file

from mpol import coordinates, gridding

# We need a fixture which provides mock visibilities of the sort we'd
# expect from visread, but *without* the CASA dependency.

# fixture to provide tuple of uu, vv, weight, data_re, and data_im values
@pytest.fixture(scope="session")
def mock_visibility_archive():

    # use astropy routines to cache data
    fname = download_file(
        "https://zenodo.org/record/4930016/files/logo_cube.noise.npz",
        cache=True,
        pkgname="mpol",
    )

    return np.load(fname)


@pytest.fixture
def mock_visibility_data(mock_visibility_archive):
    d = mock_visibility_archive
    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data = d["data"]
    data_re = np.real(data)
    data_im = np.imag(data)  # MPoL convention

    return uu, vv, weight, data_re, data_im


@pytest.fixture
def mock_visibility_data_cont(mock_visibility_archive):
    chan = 4
    d = mock_visibility_archive
    uu = d["uu"][chan]
    vv = d["vv"][chan]
    weight = d["weight"][chan]
    data = d["data"][chan]
    data_re = np.real(data)
    data_im = np.imag(data)  # MPoL convention

    return uu, vv, weight, data_re, data_im


@pytest.fixture
def coords():
    return coordinates.GridCoords(cell_size=0.005, npix=800)


@pytest.fixture
def gridder(mock_visibility_data, coords):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    gridder = gridding.Gridder(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    return gridder
    

@pytest.fixture
def dataset(mock_visibility_data, coords):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    gridder = gridding.Gridder(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    return gridder.to_pytorch_dataset()


@pytest.fixture
def dataset_cont(mock_visibility_data_cont, coords):

    uu, vv, weight, data_re, data_im = mock_visibility_data_cont
    gridder = gridding.Gridder(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    return gridder.to_pytorch_dataset()


@pytest.fixture
def crossvalidation_products(mock_visibility_data):
    # test the crossvalidation with a smaller set of image / Fourier coordinates than normal,
    # which are better matched to the extremes of the mock dataset
    coords = coordinates.GridCoords(cell_size=0.04, npix=256)

    uu, vv, weight, data_re, data_im = mock_visibility_data

    gridder = gridding.Gridder(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    dataset = gridder.to_pytorch_dataset()

    return coords, dataset


@pytest.fixture
def generic_parameters():
    # generic model parameters to test training loop and cross-val loop
    train_pars = {"epochs":50, "convergence_tol":1e-2, 
                "lambda_guess": ["entropy", "sparsity", "TV", "TSV"],
                "lambda_guess_briggs": [0.0, 0.5],
                "lambda_entropy":1e-3, "lambda_sparsity":1e-3, "lambda_TV":1e-3,
                "lambda_TSV":1e-3,"entropy_prior_intensity":1e-10, 
                "TV_epsilon":1e-10, "train_diag_step":25, 
                "diag_fig_train":False, "verbose":True     
    }
    
    crossval_pars = train_pars.copy()
    crossval_pars["learn_rate"] = 0.5
    crossval_pars["kfolds"] = 2
    crossval_pars["split_method"] = 'random_cell'
    crossval_pars["seed"] = 47
    crossval_pars["device"] = None

    gen_pars  = { "train_pars":train_pars, "crossval_pars":crossval_pars}

    return gen_pars 