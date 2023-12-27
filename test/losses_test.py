import numpy as np
import pytest
import torch

from mpol import fourier, images, losses, utils


# create a fixture that returns nchan and an image
@pytest.fixture
def nchan_cube(mock_visibility_data, coords):
    uu, vv, weight, data_re, data_im = mock_visibility_data
    nchan = len(uu)

    # create a mock base image
    basecube = images.BaseCube(
        coords=coords,
        nchan=nchan,
    )
    # insert into ImageCube layer
    imagecube = images.ImageCube(coords=coords, nchan=nchan)
    packed_cube = imagecube(basecube())

    return nchan, packed_cube


@pytest.fixture
def loose_visibilities(mock_visibility_data, coords, nchan_cube):
    # use the NuFFT to produce model visibilities

    nchan, packed_cube = nchan_cube

    # use the coil broadcasting ability
    chan = 4

    uu, vv, weight, data_re, data_im = mock_visibility_data
    uu_chan = uu[chan]
    vv_chan = vv[chan]

    nufft = fourier.NuFFT(coords=coords, nchan=nchan)
    return nufft(packed_cube, uu_chan, vv_chan)


@pytest.fixture
def gridded_visibilities(coords, nchan_cube):
    nchan, packed_cube = nchan_cube
    # use the FourierCube to produce model visibilities
    flayer = fourier.FourierCube(coords=coords)
    return flayer(packed_cube)


def test_chi_squared_evaluation(
    loose_visibilities, mock_visibility_data, gridded_visibilities, dataset
):
    # because of the way likelihood functions are defined, we would not expect
    # the loose chi_squared or log_likelihood function to give the same answers as
    # the gridded chi_squared or log_likelihood functions. This is because the normalization
    # of likelihood functions are somewhat ill-defined, since the value of the likelihood
    # function can change based on whether you bin your data.  The normalization only really makes
    # sense in a full Bayesian setting where the evidence is computed and the normalization cancels out anyway.
    # The important thing is that the *shape* of the likelihood function is the same as parameters
    # are varied.
    # more info
    # https://stats.stackexchange.com/questions/97515/what-does-likelihood-is-only-defined-up-to-a-multiplicative-constant-of-proport?noredirect=1&lq=1

    # calculate the ungridded chi^2
    uu, vv, weight, data_re, data_im = mock_visibility_data
    data = torch.tensor(data_re + 1.0j * data_im)
    weight = torch.tensor(weight)

    chi_squared = losses.chi_squared(loose_visibilities, data, weight)
    print("loose chi_squared", chi_squared)

    # calculate the gridded chi^2
    chi_squared_gridded = losses.chi_squared_gridded(gridded_visibilities, dataset)
    print("gridded chi_squared", chi_squared_gridded)

    # it's OK that the values are different


def test_log_likelihood_evaluation(
    loose_visibilities, mock_visibility_data, gridded_visibilities, dataset
):
    # see comments under test_chi_squared_evaluation for why we don't necessarily expect these
    # to be the same between gridded and ungridded

    # calculate the ungridded log likelihood
    uu, vv, weight, data_re, data_im = mock_visibility_data
    data = torch.tensor(data_re + 1.0j * data_im)
    weight = torch.tensor(weight)

    log_like = losses.log_likelihood(loose_visibilities, data, weight)
    print("loose log_likelihood", log_like)

    # calculate the gridded log likelihood
    log_like_gridded = losses.log_likelihood_gridded(gridded_visibilities, dataset)
    print("gridded log likelihood", log_like_gridded)


def test_nll_hermitian_pairs(loose_visibilities, mock_visibility_data):
    # calculate the ungridded log likelihood
    uu, vv, weight, data_re, data_im = mock_visibility_data
    data = torch.tensor(data_re + 1.0j * data_im)
    weight = torch.tensor(weight)

    log_like = losses.nll(loose_visibilities, data, weight)
    print("loose nll", log_like)

    # calculate it with Hermitian pairs

    # expand the vectors to include complex conjugates
    uu = np.concatenate([uu, -uu], axis=1)
    vv = np.concatenate([vv, -vv], axis=1)
    loose_visibilities = torch.cat(
        [loose_visibilities, torch.conj(loose_visibilities)], axis=1
    )
    data = torch.cat([data, torch.conj(data)], axis=1)
    weight = torch.cat([weight, weight], axis=1)

    log_like = losses.nll(loose_visibilities, data, weight)
    print("loose nll w/ Hermitian", log_like)


def test_nll_evaluation(
    loose_visibilities, mock_visibility_data, gridded_visibilities, dataset
):
    # We would have expected the ungridded and gridded values to be closer than they are
    # but I suppose this comes down to how the noise is averaged within each visibility cell.
    # and the definition of degrees of freedom
    # https://arxiv.org/abs/1012.3754

    # calculate the ungridded log likelihood
    uu, vv, weight, data_re, data_im = mock_visibility_data
    data = torch.tensor(data_re + 1.0j * data_im)
    weight = torch.tensor(weight)

    log_like = losses.nll(loose_visibilities, data, weight)
    print("loose nll", log_like)

    # calculate the gridded log likelihood
    log_like_gridded = losses.nll_gridded(gridded_visibilities, dataset)
    print("gridded nll", log_like_gridded)


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
