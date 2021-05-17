import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from mpol import coordinates, gridding
from mpol.constants import *


def test_gridder_instantiate_cell_npix(mock_visibility_data):
    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data_re = d["data_re"]
    data_im = -d["data_im"]

    gridding.Gridder(
        cell_size=0.005,
        npix=800,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )


def test_gridder_instantiate_gridCoord(mock_visibility_data):
    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data_re = d["data_re"]
    data_im = -d["data_im"]

    mycoords = coordinates.GridCoords(cell_size=0.005, npix=800)

    gridding.Gridder(
        coords=mycoords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )


def test_gridder_instantiate_npix_gridCoord_conflict(mock_visibility_data):
    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data_re = d["data_re"]
    data_im = -d["data_im"]

    mycoords = coordinates.GridCoords(cell_size=0.005, npix=800)

    with pytest.raises(AssertionError):
        gridding.Gridder(
            cell_size=0.005,
            npix=800,
            coords=mycoords,
            uu=uu,
            vv=vv,
            weight=weight,
            data_re=data_re,
            data_im=data_im,
        )


def test_gridder_instantiate_bounds_fail(mock_visibility_data):
    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data_re = d["data_re"]
    data_im = -d["data_im"]

    mycoords = coordinates.GridCoords(cell_size=0.05, npix=800)

    with pytest.raises(AssertionError):
        gridding.Gridder(
            coords=mycoords,
            uu=uu,
            vv=vv,
            weight=weight,
            data_re=data_re,
            data_im=data_im,
        )


# test that we're getting the right numbers back for some well defined operations
def test_uniform_ones(mock_visibility_data, tmp_path):
    coords = coordinates.GridCoords(cell_size=0.005, npix=800)

    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]
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
        gridder.sky_vis_gridded[4].real, origin="lower", extent=gridder.coords.vis_ext
    )
    plt.colorbar(im)
    plt.savefig(tmp_path / "gridded_re.png", dpi=300)

    plt.close("all")


# now that we've tested the creation ops, cache an instantiated gridder for future ops
@pytest.fixture
def gridder(mock_visibility_data):
    d = mock_visibility_data

    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data_re = d["data_re"]
    data_im = -d["data_im"]

    return gridding.Gridder(
        cell_size=0.005,
        npix=800,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )


# make sure the peak of the PSF normalizes to 1 for each channel
def test_beam_normalized(gridder):
    r = -0.5
    for weighting in ["uniform", "natural", "briggs"]:
        if weighting == "briggs":
            gridder._grid_visibilities(weighting=weighting, robust=r)
        else:
            gridder._grid_visibilities(weighting=weighting)
        beam = gridder.get_dirty_beam()

        for i in range(gridder.nchan):
            assert pytest.approx(np.max(beam[i]), 1.0)


def test_beam_null(gridder, tmp_path):
    r = -0.5
    gridder._grid_visibilities(weighting="briggs", robust=r)
    beam = gridder.get_dirty_beam()
    nulled = gridder._null_dirty_beam()

    chan = 4
    fig, ax = plt.subplots(ncols=2)

    cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
    cmap.set_under("r")
    norm = matplotlib.colors.Normalize(vmin=0)

    im = ax[0].imshow(
        beam[chan],
        origin="lower",
        interpolation="none",
        extent=gridder.coords.img_ext,
        cmap=cmap,
        norm=norm,
    )
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(
        nulled[chan] - 1e-6,
        origin="lower",
        interpolation="none",
        extent=gridder.coords.img_ext,
        cmap=cmap,
        norm=norm,
    )
    plt.colorbar(im, ax=ax[1])

    fig.savefig(tmp_path / "beam_v_nulled.png", dpi=300)
    plt.close("all")


def test_beam_null_full(gridder, tmp_path):
    r = -0.5
    gridder._grid_visibilities(weighting="briggs", robust=r)
    beam = gridder.get_dirty_beam()
    nulled = gridder._null_dirty_beam(single_channel_estimate=False)

    chan = 4
    fig, ax = plt.subplots(ncols=2)

    cmap = copy.copy(matplotlib.cm.get_cmap("viridis"))
    cmap.set_under("r")
    norm = matplotlib.colors.Normalize(vmin=0)

    im = ax[0].imshow(
        beam[chan],
        origin="lower",
        interpolation="none",
        extent=gridder.coords.img_ext,
        cmap=cmap,
        norm=norm,
    )
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(
        nulled[chan] - 1e-6,
        origin="lower",
        interpolation="none",
        extent=gridder.coords.img_ext,
        cmap=cmap,
        norm=norm,
    )
    plt.colorbar(im, ax=ax[1])

    fig.savefig(tmp_path / "beam_v_nulled.png", dpi=300)
    plt.close("all")


def test_beam_area_before_beam(gridder):
    r = -0.5
    gridder._grid_visibilities(weighting="briggs", robust=r)
    area = gridder.get_dirty_beam_area()
    print(area)


# compare uniform and robust = -2.0
def test_grid_uniform(gridder, tmp_path):

    kw = {"origin": "lower", "interpolation": "none", "extent": gridder.coords.img_ext}

    chan = 4

    img_uniform, beam_uniform = gridder.get_dirty_image(weighting="uniform")

    r = -2
    img_robust, beam_robust = gridder.get_dirty_image(weighting="briggs", robust=r)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4.5))

    ax[0, 0].imshow(beam_uniform[chan], **kw)
    ax[0, 0].set_title("uniform")
    ax[1, 0].imshow(img_uniform[chan], **kw)

    ax[0, 1].imshow(beam_robust[chan], **kw)
    ax[0, 1].set_title("robust={:}".format(r))
    ax[1, 1].imshow(img_robust[chan], **kw)

    # the differences
    im = ax[0, 2].imshow(beam_uniform[chan] - beam_robust[chan], **kw)
    plt.colorbar(im, ax=ax[0, 2])
    ax[0, 2].set_title("difference")
    im = ax[1, 2].imshow(img_uniform[chan] - img_robust[chan], **kw)
    plt.colorbar(im, ax=ax[1, 2])

    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.02, bottom=0.07, top=0.94)

    fig.savefig(tmp_path / "uniform_v_robust.png", dpi=300)

    assert np.all(np.abs(beam_uniform - beam_robust) < 1e-4)
    assert np.all(np.abs(img_uniform - img_robust) < 1e-5)

    plt.close("all")


# compare uniform and robust = -2.0
def test_grid_uniform_arcsec2(gridder, tmp_path):

    kw = {"origin": "lower", "interpolation": "none", "extent": gridder.coords.img_ext}

    chan = 4
    img_uniform, beam_uniform = gridder.get_dirty_image(
        weighting="uniform", unit="Jy/arcsec^2"
    )

    r = -2
    img_robust, beam_robust = gridder.get_dirty_image(
        weighting="briggs", robust=r, unit="Jy/arcsec^2"
    )

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4.5))

    ax[0, 0].imshow(beam_uniform[chan], **kw)
    ax[0, 0].set_title("uniform")
    im = ax[1, 0].imshow(img_uniform[chan], **kw)
    plt.colorbar(im, ax=ax[1, 0])

    ax[0, 1].imshow(beam_robust[chan], **kw)
    ax[0, 1].set_title("robust={:}".format(r))
    im = ax[1, 1].imshow(img_robust[chan], **kw)
    plt.colorbar(im, ax=ax[1, 1])

    # the differences
    im = ax[0, 2].imshow(beam_uniform[chan] - beam_robust[chan], **kw)
    plt.colorbar(im, ax=ax[0, 2])
    ax[0, 2].set_title("difference")
    im = ax[1, 2].imshow(img_uniform[chan] - img_robust[chan], **kw)
    plt.colorbar(im, ax=ax[1, 2])

    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.02, bottom=0.07, top=0.94)

    fig.savefig(tmp_path / "uniform_v_robust_arcsec2.png", dpi=300)

    assert np.all(np.abs(beam_uniform - beam_robust) < 1e-4)
    assert np.all(np.abs(img_uniform - img_robust) < 1e-3)

    plt.close("all")


def test_grid_natural(gridder, tmp_path):

    kw = {"origin": "lower", "interpolation": "none", "extent": gridder.coords.img_ext}

    chan = 4

    img_natural, beam_natual = gridder.get_dirty_image(weighting="natural")

    r = 2
    img_robust, beam_robust = gridder.get_dirty_image(weighting="briggs", robust=r)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4.5))

    ax[0, 0].imshow(beam_natural[chan], **kw)
    ax[0, 0].set_title("natural")
    ax[1, 0].imshow(img_natural[chan], **kw)

    ax[0, 1].imshow(beam_robust[chan], **kw)
    ax[0, 1].set_title("robust={:}".format(r))
    ax[1, 1].imshow(img_robust[chan], **kw)

    # the differences
    im = ax[0, 2].imshow(beam_natural[chan] - beam_robust[chan], **kw)
    plt.colorbar(im, ax=ax[0, 2])
    ax[0, 2].set_title("difference")
    im = ax[1, 2].imshow(img_natural[chan] - img_robust[chan], **kw)
    plt.colorbar(im, ax=ax[1, 2])

    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.02, bottom=0.07, top=0.94)

    fig.savefig(tmp_path / "natural_v_robust.png", dpi=300)

    assert np.all(np.abs(beam_natural - beam_robust) < 1e-3)
    assert np.all(np.abs(img_natural - img_robust) < 1e-3)

    plt.close("all")


def test_grid_natural_arcsec2(gridder, tmp_path):

    kw = {"origin": "lower", "interpolation": "none", "extent": gridder.coords.img_ext}

    chan = 4

    img_natural, beam_natural = gridder.get_dirty_image(
        weighting="natural", unit="Jy/arcsec^2"
    )

    r = 2
    img_robust, beam_robust = gridder.get_dirty_image(
        weighting="briggs", robust=r, unit="Jy/arcsec^2"
    )

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4.5))

    ax[0, 0].imshow(beam_natural[chan], **kw)
    ax[0, 0].set_title("natural")
    im = ax[1, 0].imshow(img_natural[chan], **kw)
    plt.colorbar(im, ax=ax[1, 0])

    ax[0, 1].imshow(beam_robust[chan], **kw)
    ax[0, 1].set_title("robust={:}".format(r))
    im = ax[1, 1].imshow(img_robust[chan], **kw)
    plt.colorbar(im, ax=ax[1, 1])

    # the differences
    im = ax[0, 2].imshow(beam_natural[chan] - beam_robust[chan], **kw)
    plt.colorbar(im, ax=ax[0, 2])
    ax[0, 2].set_title("difference")
    im = ax[1, 2].imshow(img_natural[chan] - img_robust[chan], **kw)
    plt.colorbar(im, ax=ax[1, 2])

    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.02, bottom=0.07, top=0.94)

    fig.savefig(tmp_path / "natural_v_robust_arcsec2.png", dpi=300)

    assert np.all(np.abs(beam_natural - beam_robust) < 1e-3)
    assert np.all(np.abs(img_natural - img_robust) < 5e-3)

    plt.close("all")


def test_weight_gridding(mock_visibility_data, tmp_path):
    d = mock_visibility_data

    uu = d["uu"]
    vv = d["vv"]
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

    # plot a histogram of weight values--should be integers.
    fig, ax = plt.subplots(nrows=1)
    ax.hist(np.log10(gridder.weight_gridded[gridder.mask]), density=True)
    ax.set_xlabel(r"$\log_{10}(\mathrm{weight})$")
    fig.savefig(tmp_path / "weight_hist.png", dpi=300)

    plt.close("all")


def test_pytorch_export(gridder):
    gridder.to_pytorch_dataset()


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
