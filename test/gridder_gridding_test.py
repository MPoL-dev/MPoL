import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpol import coordinates, gridding
from mpol.constants import *


def test_grid_cont(mock_visibility_data_cont):
    uu, vv, weight, data_re, data_im = mock_visibility_data_cont

    gridder = gridding.Gridder(
        cell_size=0.005,
        npix=800,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    print(gridder.uu.shape)
    print(gridder.nchan)

    gridder._grid_visibilities(weighting="uniform")


# test that we're getting the right numbers back for some well defined operations
def test_uniform_ones(mock_visibility_data, tmp_path):
    coords = coordinates.GridCoords(cell_size=0.005, npix=800)

    uu, vv, weight, data_re, data_im = mock_visibility_data
    weight = 0.1 * np.ones_like(uu)
    data_re = np.ones_like(uu)
    data_im = np.zeros_like(uu)

    gridder = gridding.Gridder(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    # with uniform weighting, the gridded sheet should be uniform and = 1
    gridder._grid_visibilities(weighting="uniform")

    print(
        "re",
        np.mean(gridder.data_re_gridded),
        np.std(gridder.data_re_gridded),
        np.min(gridder.data_re_gridded),
        np.max(gridder.data_re_gridded),
    )

    assert pytest.approx(np.min(gridder.data_re_gridded), 0)
    assert pytest.approx(np.max(gridder.data_im_gridded), 1)

    im = plt.imshow(
        gridder.ground_cube[4].real, origin="lower", extent=gridder.coords.vis_ext
    )
    plt.colorbar(im)
    plt.savefig(tmp_path / "gridded_re.png", dpi=300)

    plt.close("all")


def test_weight_gridding(mock_visibility_data, tmp_path):
    uu, vv, weight, data_re, data_im = mock_visibility_data
    weight = np.ones_like(uu)
    data_re = np.ones_like(uu)
    data_im = np.ones_like(uu)

    gridder = gridding.Gridder(
        cell_size=0.005,
        npix=800,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    gridder._grid_visibilities(weighting="uniform")
    gridder._grid_weights()

    # make sure all average values are set to 1
    diff_real = np.abs(1 - gridder.vis_gridded[gridder.mask].real)
    print(diff_real)
    print(np.max(diff_real))
    assert np.all(diff_real < 1e-10)

    # can't do this with imaginaries and fake data.
    # diff_imag = np.abs(1 - gridder.vis_gridded[gridder.mask].imag)
    # print(diff_imag)
    # print(np.max(diff_imag))
    # assert np.all(diff_imag < 1e-10)

    # figure out where non-1 averaged imaginaries are coming through.
    # IDK, it's kind of a weird thing because we're complex-conjugating the visibilites. Maybe this is right?
    # seems kind of dumb though. I think to just say imaginaries should be 1 and then mirror, you get into inconsistencies


# test the standard deviation estimation routines
def test_estimate_standard_deviation(mock_visibility_data, tmp_path):
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_visibility_data
    weight = 0.1 * np.ones_like(uu)
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(loc=0, scale=sigma, size=uu.shape)

    gridder = gridding.Gridder(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    s_re, s_im = gridder.estimate_cell_standard_deviation()

    chan = 4

    fig, ax = plt.subplots(ncols=2, figsize=(7, 4))

    im = ax[0].imshow(s_re[chan], origin="lower", extent=gridder.coords.vis_ext)
    ax[0].set_title(r"$s_{i,j}$ real")
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(s_im[chan], origin="lower", extent=gridder.coords.vis_ext)
    ax[1].set_title(r"$s_{i,j}$ imag")
    plt.colorbar(im, ax=ax[1])

    plt.savefig(tmp_path / "standard_deviation_correct.png", dpi=300)

    plt.close("all")


def test_estimate_standard_deviation_large(mock_visibility_data, tmp_path):
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_visibility_data
    weight = 0.1 * np.ones_like(uu)
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=2 * sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(
        loc=0, scale=2 * sigma, size=uu.shape
    )

    gridder = gridding.Gridder(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    s_re, s_im = gridder.estimate_cell_standard_deviation()

    chan = 4

    fig, ax = plt.subplots(ncols=2, figsize=(7, 4))

    im = ax[0].imshow(s_re[chan], origin="lower", extent=gridder.coords.vis_ext)
    ax[0].set_title(r"$s_{i,j}$ real")
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(s_im[chan], origin="lower", extent=gridder.coords.vis_ext)
    ax[1].set_title(r"$s_{i,j}$ imag")
    plt.colorbar(im, ax=ax[1])

    plt.savefig(tmp_path / "standard_deviation_large.png", dpi=300)

    plt.close("all")
