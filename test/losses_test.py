import pytest
import torch

from mpol import losses


def test_nll_1D_zero():
    # make fake pytorch arrays
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
    # make fake pytorch arrays
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
    # make fake pytorch arrays
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
    # make fake pytorch arrays
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
    nchan = 50
    npix = 512

    cube = torch.ones((nchan, npix, npix), dtype=torch.float64)
    losses.entropy(cube, 0.01)


def test_tsv():
    cube = torch.rand((1, 3, 3))
    tsv_val = losses.TSV(cube)
    for_val = 0
    for i in range(2):
        for j in range(2):
            for_val += (cube[:, i+1, j] - cube[:, i, j]) ** 2 + (cube[:, i, j+1] - cube[:, i, j]) ** 2
    assert tsv_val == for_val


def test_tv_image():
    cube = torch.rand((1, 3, 3))
    tsv_val = losses.TV_image(cube, epsilon=0)
    for_val = 0
    for i in range(2):
        for j in range(2):
            for_val += torch.sqrt((cube[:, i+1, j] - cube[:, i, j]) ** 2 + (cube[:, i, j+1] - cube[:, i, j]) ** 2)
    assert tsv_val == for_val
