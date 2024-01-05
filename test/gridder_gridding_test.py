import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpol import coordinates, gridding
from mpol.constants import *


def test_average_cont(coords, mock_dataset_np):
    """
    Test that the gridding operation doesn't error if provided 'continuum'-like
    quantities (single-channel).
    """
    uu, vv, weight, data_re, data_im = mock_dataset_np

    chan = 0

    averager = gridding.DataAverager(
        coords=coords,
        uu=uu[chan],
        vv=vv[chan],
        weight=weight[chan],
        data_re=data_re[chan],
        data_im=data_im[chan],
    )

    print(averager.uu.shape)
    print(averager.nchan)

    averager._grid_visibilities()


# test that we're getting the right numbers back for some well defined operations
def test_uniform_ones(mock_dataset_np, tmp_path):
    """
    Test that we can grid average a set of visibilities that are just 1.
    We should get back entirely 1s.
    """

    coords = coordinates.GridCoords(cell_size=0.005, npix=800)

    uu, vv, weight, data_re, data_im = mock_dataset_np
    weight = 0.1 * np.ones_like(uu)
    data_re = np.ones_like(uu)
    data_im = np.zeros_like(uu)

    averager = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    # with uniform weighting, the gridded values should be == 1
    averager._grid_visibilities()

    im = plt.imshow(
        averager.ground_cube[1].real,
        origin="lower",
        extent=averager.coords.vis_ext,
        interpolation="none",
    )
    plt.colorbar(im)
    plt.savefig(tmp_path / "gridded_re.png", dpi=300)

    plt.figure()

    im2 = plt.imshow(
        averager.ground_cube[0].imag,
        origin="lower",
        extent=averager.coords.vis_ext,
        interpolation="none",
    )
    plt.colorbar(im2)
    plt.savefig(tmp_path / "gridded_im.png", dpi=300)

    plt.close("all")

    # if the gridding worked,
    # cells with no data should be 0
    assert averager.data_re_gridded[~averager.mask] == pytest.approx(0)

    # and cells with data should have real values approximately 1
    assert averager.data_re_gridded[averager.mask] == pytest.approx(1)

    # and imaginary values approximately 0 everywhere
    assert averager.data_im_gridded == pytest.approx(0)


def test_weight_gridding(mock_dataset_np):
    uu, vv, weight, data_re, data_im = mock_dataset_np

    # initialize random (positive) weight values
    weight = np.random.uniform(low=0.01, high=0.1, size=uu.shape)
    data_re = np.ones_like(uu)
    data_im = np.ones_like(uu)

    coords = coordinates.GridCoords(cell_size=0.005, npix=800)
    averager = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    averager._grid_weights()

    print("sum of ungridded weights", np.sum(weight))

    # test that the weights all sum to the same value
    print("sum of gridded weights", np.sum(averager.weight_gridded))

    assert np.sum(weight) == pytest.approx(np.sum(averager.weight_gridded))


# test the standard deviation estimation routines
def test_estimate_stddev(mock_dataset_np, tmp_path):
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_dataset_np
    weight = 0.1 * np.ones_like(uu)
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(loc=0, scale=sigma, size=uu.shape)

    averager = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    s_re, s_im = averager._estimate_cell_standard_deviation()

    chan = 0

    fig, ax = plt.subplots(ncols=2, figsize=(7, 4))

    im = ax[0].imshow(s_re[chan], origin="lower", extent=averager.coords.vis_ext)
    ax[0].set_title(r"$s_{i,j}$ real")
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(s_im[chan], origin="lower", extent=averager.coords.vis_ext)
    ax[1].set_title(r"$s_{i,j}$ imag")
    plt.colorbar(im, ax=ax[1])

    plt.savefig(tmp_path / "stddev_correct.png", dpi=300)

    plt.close("all")


def test_estimate_stddev_large(mock_dataset_np, tmp_path):
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_dataset_np
    weight = 0.1 * np.ones_like(uu)
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=2 * sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(
        loc=0, scale=2 * sigma, size=uu.shape
    )

    averager = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    s_re, s_im = averager._estimate_cell_standard_deviation()

    chan = 0

    fig, ax = plt.subplots(ncols=2, figsize=(7, 4))

    im = ax[0].imshow(s_re[chan], origin="lower", extent=averager.coords.vis_ext)
    ax[0].set_title(r"$s_{i,j}$ real")
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(s_im[chan], origin="lower", extent=averager.coords.vis_ext)
    ax[1].set_title(r"$s_{i,j}$ imag")
    plt.colorbar(im, ax=ax[1])

    plt.savefig(tmp_path / "stddev_large.png", dpi=300)

    plt.close("all")


def test_max_scatter_pass(mock_dataset_np):
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_dataset_np
    weight = 0.1 * np.ones_like(uu)
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(loc=0, scale=sigma, size=uu.shape)

    averager = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    # we want this to return an exit code of True, indicating an error
    d = averager._check_scatter_error()
    print(d["median_re"], d["median_im"])
    assert not d["return_status"]


def test_max_scatter_fail(mock_dataset_np):
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_dataset_np
    weight = 0.1 * np.ones_like(uu)
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=2 * sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(
        loc=0, scale=2 * sigma, size=uu.shape
    )

    averager = gridding.DataAverager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    # we want this to return an exit code of True, indicating an error
    d = averager._check_scatter_error()
    print(d["median_re"], d["median_im"])
    assert d["return_status"]
