import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpol import coordinates, gridding
from mpol.constants import *


def test_grid_cont(mock_visibility_data_cont):
    """
    Test that the gridding operation doesn't error.
    """
    uu, vv, weight, data_re, data_im = mock_visibility_data_cont

    dataavg = gridding.DataAverager.from_image_properties(
        cell_size=0.005,
        npix=800,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    print(dataavg.uu.shape)
    print(dataavg.nchan)

    dataavg._grid_visibilities(weighting="uniform")


# test that we're getting the right numbers back for some well defined operations
def test_uniform_ones(mock_visibility_data, tmp_path):
    """
    Test that we can grid average a set of visibilities that are just 1.
    We should get back entirely 1s.
    """

    coords = coordinates.GridCoords(cell_size=0.005, npix=800)

    uu, vv, weight, data_re, data_im = mock_visibility_data
    weight = 0.1 * np.ones_like(uu)
    data_re = np.ones_like(uu)
    data_im = np.zeros_like(uu)

    dataavg = gridding.datavg(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    # with uniform weighting, the gridded sheet should be uniform and = 1
    dataavg._grid_visibilities(weighting="uniform")

    print(
        "re",
        np.mean(dataavg.data_re_gridded),
        np.std(dataavg.data_re_gridded),
        np.min(dataavg.data_re_gridded),
        np.max(dataavg.data_re_gridded),
    )

    im = plt.imshow(
        dataavg.ground_cube[4].real, origin="lower", extent=dataavg.coords.vis_ext
    )
    plt.colorbar(im)
    plt.savefig(tmp_path / "gridded_re.png", dpi=300)

    plt.figure()

    im2 = plt.imshow(
        dataavg.ground_cube[4].imag, origin="lower", extent=dataavg.coords.vis_ext
    )
    plt.colorbar(im2)
    plt.savefig(tmp_path / "gridded_im.png", dpi=300)

    plt.close("all")

    # if the gridding worked, we should have real values approximately 1
    assert np.max(dataavg.data_re_gridded) == pytest.approx(1)
    # except in the cells with no data
    assert np.min(dataavg.data_re_gridded) == pytest.approx(0)

    # make sure all average values are set to 1
    diff_real = np.abs(1 - dataavg.vis_gridded[dataavg.mask].real)
    assert np.all(diff_real < 1e-10)

    # and imaginary values approximately 0 everywhere
    assert np.min(dataavg.data_im_gridded) == pytest.approx(0)
    assert np.max(dataavg.data_im_gridded) == pytest.approx(0)


def test_weight_gridding(mock_visibility_data):
    uu, vv, weight, data_re, data_im = mock_visibility_data

    # initialize random (positive) weight values
    weight = np.random.uniform(low=0.01, high=0.1, size=uu.shape)
    data_re = np.ones_like(uu)
    data_im = np.ones_like(uu)

    dataavg = gridding.DataAverager.from_image_properties(
        cell_size=0.005,
        npix=800,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    dataavg._grid_weights()

    print("sum of ungridded weights", np.sum(weight))

    # test that the weights all sum to the same value, modulo Hermitian aspects
    # should be twice that of the ungridded weights, since Hermitian weights have
    # been double-counted
    print("sum of gridded weights", np.sum(dataavg.weight_gridded))

    assert np.sum(weight) == pytest.approx(0.5 * np.sum(dataavg.weight_gridded))


# test the standard deviation estimation routines
def test_estimate_stddev(mock_visibility_data, tmp_path):
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_visibility_data
    weight = 0.1 * np.ones_like(uu)
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(loc=0, scale=sigma, size=uu.shape)

    dataavg = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    s_re, s_im = dataavg._estimate_cell_standard_deviation()

    chan = 4

    fig, ax = plt.subplots(ncols=2, figsize=(7, 4))

    im = ax[0].imshow(s_re[chan], origin="lower", extent=dataavg.coords.vis_ext)
    ax[0].set_title(r"$s_{i,j}$ real")
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(s_im[chan], origin="lower", extent=dataavg.coords.vis_ext)
    ax[1].set_title(r"$s_{i,j}$ imag")
    plt.colorbar(im, ax=ax[1])

    plt.savefig(tmp_path / "stddev_correct.png", dpi=300)

    plt.close("all")


def test_estimate_stddev_large(mock_visibility_data, tmp_path):
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_visibility_data
    weight = 0.1 * np.ones_like(uu)
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=2 * sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(
        loc=0, scale=2 * sigma, size=uu.shape
    )

    dataavg = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    s_re, s_im = dataavg._estimate_cell_standard_deviation()

    chan = 4

    fig, ax = plt.subplots(ncols=2, figsize=(7, 4))

    im = ax[0].imshow(s_re[chan], origin="lower", extent=dataavg.coords.vis_ext)
    ax[0].set_title(r"$s_{i,j}$ real")
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(s_im[chan], origin="lower", extent=dataavg.coords.vis_ext)
    ax[1].set_title(r"$s_{i,j}$ imag")
    plt.colorbar(im, ax=ax[1])

    plt.savefig(tmp_path / "stddev_large.png", dpi=300)

    plt.close("all")


def test_max_scatter_pass(mock_visibility_data):
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_visibility_data
    weight = 0.1 * np.ones_like(uu)
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(loc=0, scale=sigma, size=uu.shape)

    dataavg = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    # we want this to return an exit code of True, indicating an error
    d = dataavg._check_scatter_error()
    print(d["median_re"], d["median_im"])
    assert not d["return_status"]


def test_max_scatter_fail(mock_visibility_data):
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_visibility_data
    weight = 0.1 * np.ones_like(uu)
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=2 * sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(
        loc=0, scale=2 * sigma, size=uu.shape
    )

    dataavg = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    # we want this to return an exit code of True, indicating an error
    d = dataavg._check_scatter_error()
    print(d["median_re"], d["median_im"])
    assert d["return_status"]
