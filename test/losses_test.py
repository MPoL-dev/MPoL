import torch
import pytest

from mpol import losses


def test_nll_1D_zero():
    # make fake pytorch arrays
    N = 10000
    weights = torch.ones((N), dtype=torch.float64)

    model_re = torch.randn_like(weights)
    model_im = torch.randn_like(weights)
    model_vis = (model_re, model_im)

    data_re = model_re
    data_im = model_im
    data_vis = (data_re, data_im, weights)

    loss = losses.loss_fn(model_vis, data_vis)
    assert loss.item() == 0.0


def test_nll_1D_random():
    # make fake pytorch arrays
    N = 10000
    weights = torch.ones((N), dtype=torch.float64)

    model_re = torch.randn_like(weights)
    model_im = torch.randn_like(weights)
    model_vis = (model_re, model_im)

    data_re = torch.randn_like(weights)
    data_im = torch.randn_like(weights)
    data_vis = (data_re, data_im, weights)

    losses.loss_fn(model_vis, data_vis)


def test_nll_2D_zero():
    # sometimes thing come in as a (nchan, nvis) tensor
    # make fake pytorch arrays
    nchan = 50
    nvis = 10000
    weights = torch.ones((nchan, nvis), dtype=torch.float64)

    model_re = torch.randn_like(weights)
    model_im = torch.randn_like(weights)
    model_vis = (model_re, model_im)

    data_re = model_re
    data_im = model_im
    data_vis = (data_re, data_im, weights)

    loss = losses.loss_fn(model_vis, data_vis)
    assert loss.item() == 0.0


def test_nll_2D_random():
    # sometimes thing come in as a (nchan, nvis) tensor
    # make fake pytorch arrays
    nchan = 50
    nvis = 10000
    weights = torch.ones((nchan, nvis), dtype=torch.float64)

    model_re = torch.randn_like(weights)
    model_im = torch.randn_like(weights)
    model_vis = (model_re, model_im)

    data_re = torch.randn_like(weights)
    data_im = torch.randn_like(weights)
    data_vis = (data_re, data_im, weights)

    losses.loss_fn(model_vis, data_vis)


def test_entropy_raise_error_negative():
    nchan = 50
    npix = 512
    with pytest.raises(AssertionError):
        cube = torch.randn((nchan, npix, npix), dtype=torch.float64)
        losses.loss_fn_entropy(cube, 0.01)


def test_entropy_raise_error_negative_prior():
    nchan = 50
    npix = 512
    with pytest.raises(AssertionError):
        cube = torch.ones((nchan, npix, npix), dtype=torch.float64)
        losses.loss_fn_entropy(cube, -0.01)


def test_entropy_cube():
    nchan = 50
    npix = 512

    cube = torch.ones((nchan, npix, npix), dtype=torch.float64)
    losses.loss_fn_entropy(cube, 0.01)
