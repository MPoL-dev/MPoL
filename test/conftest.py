import numpy as np
import pytest
import torch
import visread.process
from astropy.utils.data import download_file

from mpol import coordinates, fourier, gridding, images, utils
from mpol.__init__ import zenodo_record

from importlib.resources import files

# private variables to this module
_npz_path = files("mpol.data").joinpath("mock_data.npz")
_nchan = 4
_cell_size = 0.005

# all of these are fixed quantities that could take a while to load from the
# archive, so we scope them as session


@pytest.fixture(scope="session")
def img2D_butterfly():
    """Return the 2D source image of the butterfly, for use as a test image cube."""
    archive = np.load(_npz_path)
    img = np.float64(archive["img"])

    # assuming we're going to go with _cell_size, set the total flux of this image
    # total flux should be 0.253 Jy from MPoL-examples.

    return img


@pytest.fixture(scope="session")
def packed_cube(img2D_butterfly):
    """Create a packed tensor image cube from the butterfly."""
    print("npix packed cube", img2D_butterfly.shape)
    # tile to some nchan, npix, npix
    sky_cube = torch.tile(torch.from_numpy(img2D_butterfly), (_nchan, 1, 1))
    # convert to packed format
    return utils.sky_cube_to_packed_cube(sky_cube)


@pytest.fixture(scope="session")
def baselines_m():
    "Return the mock baselines (in meters) produced from the IM Lup DSHARP dataset."
    archive = np.load(_npz_path)
    return np.float64(archive["uu"]), np.float64(archive["vv"])


@pytest.fixture(scope="session")
def baselines_1D(baselines_m):
    uu, vv = baselines_m

    # lambda for now
    uu = visread.process.convert_baselines(uu, 230.0e9)
    vv = visread.process.convert_baselines(vv, 230.0e9)

    # convert to torch
    return torch.as_tensor(uu), torch.as_tensor(vv)


@pytest.fixture(scope="session")
def baselines_2D_np(baselines_m):
    uu, vv = baselines_m

    u_lam, v_lam = visread.process.broadcast_and_convert_baselines(
        uu, vv, np.linspace(230.0, 231.0, num=_nchan) * 1e9
    )

    return u_lam, v_lam


@pytest.fixture(scope="session")
def baselines_2D_t(baselines_2D_np):
    uu, vv = baselines_2D_np
    return torch.as_tensor(uu), torch.as_tensor(vv)


@pytest.fixture(scope="session")
def weight_1D_np():
    archive = np.load(_npz_path)
    return np.float64(archive["weight"])


@pytest.fixture(scope="session")
def weight_2D_t(baselines_2D_t, weight_1D_np):
    weight1D_t = torch.as_tensor(weight_1D_np)
    uu, vv = baselines_2D_t
    weight = torch.broadcast_to(weight1D_t, uu.size())
    return weight


@pytest.fixture(scope="session")
def coords(img2D_butterfly):
    npix, _ = img2D_butterfly.shape
    # note that this is now the same as the mock image we created
    return coordinates.GridCoords(cell_size=_cell_size, npix=npix)


@pytest.fixture(scope="session")
def mock_data_t(baselines_2D_t, packed_cube, coords, weight_2D_t):
    uu, vv = baselines_2D_t
    data, _ = fourier.generate_fake_data(packed_cube, coords, uu, vv, weight_2D_t)
    return data


@pytest.fixture(scope="session")
def mock_dataset_np(baselines_2D_np, weight_2D_t, mock_data_t):
    uu, vv = baselines_2D_np
    weight = utils.torch2npy(weight_2D_t)
    data = utils.torch2npy(mock_data_t)
    data_re = np.real(data)
    data_im = np.imag(data)

    return (uu, vv, weight, data_re, data_im)


@pytest.fixture(scope="session")
def dataset(mock_dataset_np, coords):
    uu, vv, weight, data_re, data_im = mock_dataset_np

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
def dataset_cont(mock_dataset_np, coords):
    uu, vv, weight, data_re, data_im = mock_dataset_np
    # select only the 0th channel of each
    averager = gridding.DataAverager(
        coords=coords,
        uu=uu[0],
        vv=vv[0],
        weight=weight[0],
        data_re=data_re[0],
        data_im=data_im[0],
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
