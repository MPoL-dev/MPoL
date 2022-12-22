import numpy as np
import pytest
import torch

from mpol import fourier, images, losses, utils


# create a fixture that has an image and produces loose and gridded model visibilities
@pytest.fixture
def image_cube(mock_visibility_data, coords):
    # Gaussian parameters
    kw = {
        "a": 1,
        "delta_x": 0.02,  # arcsec
        "delta_y": -0.01,
        "sigma_x": 0.02,
        "sigma_y": 0.01,
        "Omega": 20,  # degrees
    }

    uu, vv, weight, data_re, data_im = mock_visibility_data
    nchan = len(uu)

    # evaluate the Gaussian over the sky-plane, as np array
    img_packed = utils.sky_gaussian_arcsec(
        coords.packed_x_centers_2D, coords.packed_y_centers_2D, **kw
    )

    # broadcast to (nchan, npix, npix)
    img_packed_cube = np.broadcast_to(
        img_packed, (nchan, coords.npix, coords.npix)
    ).copy()
    # convert img_packed to pytorch tensor
    img_packed_tensor = torch.from_numpy(img_packed_cube)
    # insert into ImageCube layer

    return images.ImageCube(coords=coords, nchan=nchan, cube=img_packed_tensor)


@pytest.fixture
def loose_visibilities(mock_visibility_data, image_cube):
    # use the NuFFT to produce model visibilities

    nchan = image_cube.nchan

    # use the coil broadcasting ability
    chan = 4

    uu, vv, weight, data_re, data_im = mock_visibility_data
    uu_chan = uu[chan]
    vv_chan = vv[chan]

    nufft = fourier.NuFFT(coords=image_cube.coords, nchan=nchan, uu=uu_chan, vv=vv_chan)
    return nufft.forward(image_cube.forward())


@pytest.fixture
def gridded_visibilities(image_cube):

    # use the FourierCube to produce model visibilities
    flayer = fourier.FourierCube(coords=image_cube.coords)
    return flayer.forward(image_cube.forward())


def test_chi_squared_comparison(loose_visibilities, mock_visibility_data):
    # Compare the chi squared values from the loose and gridded visibilities

    # calculate the ungridded chi^2
    uu, vv, weight, data_re, data_im = mock_visibility_data
    data = torch.tensor(data_re + 1.0j * data_im)
    weight = torch.tensor(weight)

    chi_squared = losses.chi_squared(loose_visibilities, data, weight)

    # calculate the gridded chi^2


def test_chi_squared_comp_gridded(loose_visibilities):
    pass


def test_nll_1D_zero():
    # make identical fake pytorch arrays for data and model
    # assert that nll losses returns 0

    N = 10000
    weights = torch.ones((N), dtype=torch.float64)

    model_re = torch.randn_like(weights)
    model_im = torch.randn_like(weights)
    model_vis = torch.complex(model_re, model_im)

    data_re = model_re
    data_im = model_im
    data_vis = torch.complex(data_re, data_im)

    loss = losses.nll(model_vis, data_vis, weights)
    assert loss.item() == 0.0


def test_nll_1D_random():
    # make fake pytorch arrays that are random
    # and then test that the nll version evaluates

    N = 10000
    weights = torch.ones((N), dtype=torch.float64)

    model_re = torch.randn_like(weights)
    model_im = torch.randn_like(weights)
    model_vis = torch.complex(model_re, model_im)

    data_re = torch.randn_like(weights)
    data_im = torch.randn_like(weights)
    data_vis = torch.complex(data_re, data_im)

    losses.nll(model_vis, data_vis, weights)


def test_nll_2D_zero():
    # sometimes thing come in as a (nchan, nvis) tensor
    # make identical fake pytorch arrays in this size,
    # and assert that they evaluate the same

    nchan = 50
    nvis = 10000
    weights = torch.ones((nchan, nvis), dtype=torch.float64)

    model_re = torch.randn_like(weights)
    model_im = torch.randn_like(weights)
    model_vis = torch.complex(model_re, model_im)

    data_re = model_re
    data_im = model_im
    data_vis = torch.complex(data_re, data_im)

    loss = losses.nll(model_vis, data_vis, weights)
    assert loss.item() == 0.0


def test_nll_2D_random():
    # sometimes thing come in as a (nchan, nvis) tensor
    # make random fake pytorch arrays and make sure we can evaluate the function

    nchan = 50
    nvis = 10000
    weights = torch.ones((nchan, nvis), dtype=torch.float64)

    model_re = torch.randn_like(weights)
    model_im = torch.randn_like(weights)
    model_vis = torch.complex(model_re, model_im)

    data_re = torch.randn_like(weights)
    data_im = torch.randn_like(weights)
    data_vis = torch.complex(data_re, data_im)

    losses.nll(model_vis, data_vis, weights)


def test_entropy_raise_error_negative():
    nchan = 50
    npix = 512
    with pytest.raises(AssertionError):
        cube = torch.randn((nchan, npix, npix), dtype=torch.float64)
        losses.entropy(cube, 0.01)


def test_entropy_raise_error_negative_prior():
    nchan = 50
    npix = 512
    with pytest.raises(AssertionError):
        cube = torch.ones((nchan, npix, npix), dtype=torch.float64)
        losses.entropy(cube, -0.01)


def test_entropy_cube():
    # make a cube that should evaluate within the entropy loss function

    nchan = 50
    npix = 512

    cube = torch.ones((nchan, npix, npix), dtype=torch.float64)
    losses.entropy(cube, 0.01)


def test_tsv():
    # Here we test the accuracy of the losses.TSV() routine relative to what is
    # written in equations. Since for-loops in python are typically slow, it is
    # unreasonable to use this format in the TSV() function so a vector math format is used.
    # Here we test to ensure that this vector math is calculates correctly and results in the
    # same value as would come from the for-loop.

    # setting the size of our image
    npix = 3

    # creating the test cube
    cube = torch.rand((1, npix, npix))

    # finding the value that our TSV function returns
    tsv_val = losses.TSV(cube)

    for_val = 0
    # calculating the TSV loss through a for loop
    for i in range(npix - 1):
        for j in range(npix - 1):
            for_val += (cube[:, i + 1, j] - cube[:, i, j]) ** 2 + (
                cube[:, i, j + 1] - cube[:, i, j]
            ) ** 2
    # asserting that these two values calculated above are equivalent
    assert tsv_val == for_val


def test_tv_image():
    # Here we test the losses.TV_image(). Since for-loops in python are typically slow, it is
    # unreasonable to use this format in the TV_image() function so a vector math format is used.
    # Here we test to ensure that this vector math is calculates correctly and results in the same
    # value as would come from the for-loop.

    # setting the size of our image
    npix = 3

    # creating the test cube
    cube = torch.rand((1, npix, npix))

    # finding the value that our TV_image function returns, we set epsilon=0 for a simpler for-loop
    tsv_val = losses.TV_image(cube, epsilon=0)
    for_val = 0

    # calculating the TV_image loss through a for loop
    for i in range(npix - 1):
        for j in range(npix - 1):
            for_val += torch.sqrt(
                (cube[:, i + 1, j] - cube[:, i, j]) ** 2
                + (cube[:, i, j + 1] - cube[:, i, j]) ** 2
            )
    # asserting that these two values calculated above are equivalent
    assert tsv_val == for_val
