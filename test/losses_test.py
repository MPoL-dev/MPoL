import numpy as np
import pytest
import torch
from mpol import coordinates, fourier, losses


@pytest.fixture
def loose_vis_model(weight_2D_t):
    # just random noise is fine for these structural tests
    mean = torch.zeros_like(weight_2D_t)
    sigma = torch.sqrt(1 / weight_2D_t)
    model_re = torch.normal(mean, sigma)
    model_im = torch.normal(mean, sigma)
    return torch.complex(model_re, model_im)


@pytest.fixture
def gridded_vis_model(coords, packed_cube):
    nchan, npix, _ = packed_cube.size()
    coords = coordinates.GridCoords(npix=npix, cell_size=0.005)

    # use the FourierCube to produce model visibilities
    flayer = fourier.FourierCube(coords=coords)
    return flayer(packed_cube)


def test_chi_squared_evaluation(
    loose_vis_model, mock_data_t, weight_2D_t
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
    chi_squared = losses._chi_squared(loose_vis_model, mock_data_t, weight_2D_t)
    print("loose chi_squared", chi_squared)


def test_log_likelihood_loose(loose_vis_model, mock_data_t, weight_2D_t):
    # calculate the ungridded log likelihood
    losses.log_likelihood(loose_vis_model, mock_data_t, weight_2D_t)

def test_log_likelihood_gridded(gridded_vis_model, dataset):
    losses.log_likelihood_gridded(gridded_vis_model, dataset)


def test_rchi_evaluation(
    loose_vis_model, mock_data_t, weight_2D_t, gridded_vis_model, dataset
):
    # We would have expected the ungridded and gridded values to be closer than they are
    # but I suppose this comes down to how the noise is averaged within each visibility cell.
    # and the definition of degrees of freedom
    # https://arxiv.org/abs/1012.3754

    # calculate the ungridded log likelihood
    log_like = losses.r_chi_squared(loose_vis_model, mock_data_t, weight_2D_t)
    print("loose nll", log_like)

    # calculate the gridded log likelihood
    print(gridded_vis_model.size(), dataset.mask.size())
    log_like_gridded = losses.r_chi_squared_gridded(gridded_vis_model, dataset)
    print("gridded nll", log_like_gridded)


def test_r_chi_1D_zero():
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

    loss = losses.r_chi_squared(model_vis, data_vis, weights)
    assert loss.item() == 0.0


def test_r_chi_1D_random():
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

    losses.r_chi_squared(model_vis, data_vis, weights)


def test_r_chi_2D_zero():
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

    loss = losses.r_chi_squared(model_vis, data_vis, weights)
    assert loss.item() == 0.0


def test_r_chi_2D_random():
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

    losses.r_chi_squared(model_vis, data_vis, weights)


def test_loss_scaling():
    for N in np.logspace(4, 5, num=10):
        # create fake model, resid, and weight
        N = int(N)

        mean = torch.zeros(N)
        std = 0.2 * torch.ones(N)
        weight = 1 / std**2

        model_real = torch.ones(N)
        model_imag = torch.zeros(N)
        model = torch.complex(model_real, model_imag)

        noise_real = torch.normal(mean, std)
        noise_imag = torch.normal(mean, std)
        noise = torch.complex(noise_real, noise_imag)

        data = model + noise

        nlla = losses.neg_log_likelihood_avg(model, data, weight)
        print("N", N, "nlla", nlla)


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
