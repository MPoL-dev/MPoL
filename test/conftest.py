import numpy as np
import pytest
import torch
import visread.process
from astropy.utils.data import download_file

from mpol import coordinates, fourier, gridding, images, utils
from mpol.__init__ import zenodo_record

from importlib.resources import files

npz_path = files("mpol.data").joinpath("mock_data.npz")


# fixture to provide tuple of uu, vv, weight, data_re, and data_im values
@pytest.fixture(scope="session")
def mock_visibility_archive():
    # use astropy routines to cache data
    fname = download_file(
        f"https://zenodo.org/record/{zenodo_record}/files/logo_cube.noise.npz",
        cache=True,
        pkgname="mpol",
    )

    return np.load(fname)


@pytest.fixture(scope="session")
def img2D_butterfly():
    """Return the 2D source image of the butterfly, for use as a test image cube."""
    archive = np.load(npz_path)
    return np.float64(archive["img"])


@pytest.fixture(scope="session")
def packed_cube(img2D_butterfly):
    """Create a packed tensor image cube from the butterfly."""
    # now (1, npix, npix)
    print("npix packed cube", img2D_butterfly.shape)
    nchan = 10
    # tile to some nchan, npix, npix
    cube = torch.tile(torch.from_numpy(img2D_butterfly), (nchan, 1, 1))
    # convert to packed format
    return utils.sky_cube_to_packed_cube(cube)


@pytest.fixture(scope="session")
def baselines_m():
    "Return the mock baselines (in meters) produced from the IM Lup DSHARP dataset."
    archive = np.load(npz_path)
    return np.float64(archive["uu"]), np.float64(archive["vv"])


@pytest.fixture
def baselines_1D(baselines_m):
    uu, vv = baselines_m

    # klambda for now
    return 1e-3 * visread.process.convert_baselines(
        uu, 230.0e9
    ), 1e-3 * visread.process.convert_baselines(vv, 230.0e9)


@pytest.fixture
def baselines_2D(baselines_m):
    uu, vv = baselines_m

    u_lam, v_lam = visread.process.broadcast_and_convert_baselines(
        uu, vv, np.array([230.0, 230.01, 230.02]) * 1e9
    )

    # klambda for now
    return 1e-3 * u_lam, 1e-3 * v_lam


# to replace everything with the mock dataset (and pass), we need to replace
# * mock_visibility_data
# * mock_visibility_data_cont
#
# audit of test suite usage. Routines require
# * all uu, vv, weight, data_re, data_im, single-channel
# * all uu, vv, weight, data_re, data_im, multi-channel


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
    # note that this is now the same as the mock image we created
    return coordinates.GridCoords(cell_size=0.005, npix=1028)


@pytest.fixture
def averager(mock_visibility_data, coords):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    averager = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    return averager


@pytest.fixture
def imager(mock_visibility_data, coords):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    imager = gridding.DirtyImager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    return imager


@pytest.fixture
def dataset(mock_visibility_data, coords):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    averager = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    return averager.to_pytorch_dataset()


@pytest.fixture
def dataset_cont(mock_visibility_data_cont, coords):
    uu, vv, weight, data_re, data_im = mock_visibility_data_cont
    averager = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    return averager.to_pytorch_dataset()


@pytest.fixture(scope="session")
def mock_1d_archive():
    # use astropy routines to cache data
    fname = download_file(
        f"https://zenodo.org/record/{zenodo_record}/files/mock_disk_1d.npz",
        cache=True,
        pkgname="mpol",
    )

    return np.load(fname, allow_pickle=True)


@pytest.fixture
def mock_1d_image_model(mock_1d_archive):
    m = mock_1d_archive
    rtrue = m["rtrue"]
    itrue = m["itrue"]
    i2dtrue = m["i2dtrue"]
    xmax = ymax = m["xmax"]
    geom = m["geometry"]
    geom = geom[()]

    coords = coordinates.GridCoords(
        cell_size=xmax * 2 / i2dtrue.shape[0], npix=i2dtrue.shape[0]
    )

    # the center of the array is already at the center of the image -->
    # undo this as expected by input to ImageCube
    i2dtrue = np.flip(np.fft.fftshift(i2dtrue), 1)

    # pack the numpy image array into an ImageCube
    packed_cube = np.broadcast_to(i2dtrue, (1, coords.npix, coords.npix)).copy()
    packed_tensor = torch.from_numpy(packed_cube)
    bcube = images.BaseCube(
        coords=coords, nchan=1, base_cube=packed_tensor, pixel_mapping=lambda x: x
    )
    cube_true = images.ImageCube(coords=coords, nchan=1)
    # register cube to buffer inside cube_true.cube
    cube_true(bcube())

    return rtrue, itrue, cube_true, xmax, ymax, geom


@pytest.fixture
def mock_1d_vis_model(mock_1d_archive):
    m = mock_1d_archive
    i2dtrue = m["i2dtrue"]
    xmax = m["xmax"]
    geom = m["geometry"]
    geom = geom[()]

    Vtrue = m["vis"]
    Vtrue_dep = m["vis_dep"]
    q_dep = m["baselines_dep"]

    coords = coordinates.GridCoords(
        cell_size=xmax * 2 / i2dtrue.shape[0], npix=i2dtrue.shape[0]
    )

    # the center of the array is already at the center of the image -->
    # undo this as expected by input to ImageCube
    i2dtrue = np.flip(np.fft.fftshift(i2dtrue), 1)

    # pack the numpy image array into an ImageCube
    packed_cube = np.broadcast_to(i2dtrue, (1, coords.npix, coords.npix)).copy()
    packed_tensor = torch.from_numpy(packed_cube)
    bcube = images.BaseCube(
        coords=coords, nchan=1, base_cube=packed_tensor, pixel_mapping=lambda x: x
    )
    cube_true = images.ImageCube(coords=coords, nchan=1)

    # register image
    cube_true(bcube())

    # create a FourierCube
    fcube_true = fourier.FourierCube(coords=coords)

    # take FT of icube to populate fcube
    fcube_true.forward(cube_true.sky_cube)

    # insert the vis tensor into the FourierCube ('vis' would typically be
    # populated by taking the FFT of an image)
    # packed_fcube = np.broadcast_to(Vtrue, (1, len(Vtrue))).copy()
    # packed_ftensor = torch.from_numpy(packed_cube)
    # fcube_true.ground_cube = packed_tensor

    return fcube_true, Vtrue_dep, q_dep, geom


@pytest.fixture
def crossvalidation_products(mock_visibility_data):
    # test the crossvalidation with a smaller set of image / Fourier coordinates than normal,
    # which are better matched to the extremes of the mock dataset
    coords = coordinates.GridCoords(cell_size=0.04, npix=256)

    uu, vv, weight, data_re, data_im = mock_visibility_data

    averager = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    dataset = averager.to_pytorch_dataset()

    return coords, dataset


@pytest.fixture
def generic_parameters(tmp_path):
    # generic model parameters to test training loop and cross-val loop
    regularizers = {
        "entropy": {"lambda": 1e-3, "guess": False, "prior_intensity": 1e-10},
    }

    train_pars = {
        "epochs": 15,
        "convergence_tol": 1e-3,
        "regularizers": regularizers,
        "train_diag_step": None,
        "save_prefix": tmp_path,
        "verbose": True,
    }

    crossval_pars = train_pars.copy()
    crossval_pars["learn_rate"] = 0.5
    crossval_pars["kfolds"] = 2
    crossval_pars["split_method"] = "random_cell"
    crossval_pars["seed"] = 47
    crossval_pars["split_diag_fig"] = False
    crossval_pars["store_cv_diagnostics"] = True
    crossval_pars["device"] = None

    gen_pars = {"train_pars": train_pars, "crossval_pars": crossval_pars}

    return gen_pars
