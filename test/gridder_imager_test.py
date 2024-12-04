import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpol import coordinates, gridding


# cache an instantiated imager for future imaging ops
@pytest.fixture
def imager(mock_dataset_np, coords):
    uu, vv, weight, data_re, data_im = mock_dataset_np

    return gridding.DirtyImager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )


# make sure the peak of the PSF normalizes to 1 for each channel
def test_beam_normalized(imager):
    r = -0.5
    for weighting in ["uniform", "natural", "briggs"]:
        if weighting == "briggs":
            imager._grid_visibilities(weighting=weighting, robust=r)
        else:
            imager._grid_visibilities(weighting=weighting)
        beam = imager._get_dirty_beam(imager.C, imager.re_gridded_beam)

        for i in range(imager.nchan):
            assert np.max(beam[i]) == pytest.approx(1.0)


def test_beam_null(imager, tmp_path):
    r = -0.5
    imager._grid_visibilities(weighting="briggs", robust=r)
    beam = imager._get_dirty_beam(imager.C, imager.re_gridded_beam)
    nulled = imager._null_dirty_beam()

    chan = 0
    fig, ax = plt.subplots(ncols=2)

    cmap = copy.copy(matplotlib.colormaps["viridis"])
    cmap.set_under("r")
    norm = matplotlib.colors.Normalize(vmin=0)

    im = ax[0].imshow(
        beam[chan],
        origin="lower",
        interpolation="none",
        extent=imager.coords.img_ext,
        cmap=cmap,
        norm=norm,
    )
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(
        nulled[chan] - 1e-6,
        origin="lower",
        interpolation="none",
        extent=imager.coords.img_ext,
        cmap=cmap,
        norm=norm,
    )
    plt.colorbar(im, ax=ax[1])

    fig.savefig(tmp_path / "beam_v_nulled.png", dpi=300)
    plt.close("all")


def test_beam_null_full(imager, tmp_path):
    r = -0.5
    imager._grid_visibilities(weighting="briggs", robust=r)
    beam = imager._get_dirty_beam(imager.C, imager.re_gridded_beam)
    nulled = imager._null_dirty_beam(single_channel_estimate=False)

    chan = 0
    fig, ax = plt.subplots(ncols=2)

    cmap = copy.copy(matplotlib.colormaps["viridis"])
    cmap.set_under("r")
    norm = matplotlib.colors.Normalize(vmin=0)

    im = ax[0].imshow(
        beam[chan],
        origin="lower",
        interpolation="none",
        extent=imager.coords.img_ext,
        cmap=cmap,
        norm=norm,
    )
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(
        nulled[chan] - 1e-6,
        origin="lower",
        interpolation="none",
        extent=imager.coords.img_ext,
        cmap=cmap,
        norm=norm,
    )
    plt.colorbar(im, ax=ax[1])

    fig.savefig(tmp_path / "beam_v_nulled.png", dpi=300)
    plt.close("all")


def test_beam_area_before_beam(imager):
    r = -0.5
    imager._grid_visibilities(weighting="briggs", robust=r)
    area = imager.get_dirty_beam_area()
    print(area)


# compare uniform and robust = -2.0
def test_grid_uniform(imager, tmp_path):
    kw = {"origin": "lower", "interpolation": "none", "extent": imager.coords.img_ext}

    chan = 0

    img_uniform, beam_uniform = imager.get_dirty_image(
        weighting="uniform", check_visibility_scatter=False
    )

    r = -2
    img_robust, beam_robust = imager.get_dirty_image(
        weighting="briggs", robust=r, check_visibility_scatter=False
    )

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4.5))

    ax[0, 0].imshow(beam_uniform[chan], **kw)
    ax[0, 0].set_title("uniform")
    ax[1, 0].imshow(img_uniform[chan], **kw)

    ax[0, 1].imshow(beam_robust[chan], **kw)
    ax[0, 1].set_title(f"robust={r}")
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
    assert np.all(np.abs(img_uniform - img_robust) < 1e-4)

    plt.close("all")


# compare uniform and robust = -2.0
def test_grid_uniform_arcsec2(imager, tmp_path):
    kw = {"origin": "lower", "interpolation": "none", "extent": imager.coords.img_ext}

    chan = 0
    img_uniform, beam_uniform = imager.get_dirty_image(
        weighting="uniform", unit="Jy/arcsec^2", check_visibility_scatter=False
    )

    r = -2
    img_robust, beam_robust = imager.get_dirty_image(
        weighting="briggs", robust=r, unit="Jy/arcsec^2", check_visibility_scatter=False
    )

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4.5))

    ax[0, 0].imshow(beam_uniform[chan], **kw)
    ax[0, 0].set_title("uniform")
    im = ax[1, 0].imshow(img_uniform[chan], **kw)
    plt.colorbar(im, ax=ax[1, 0])

    ax[0, 1].imshow(beam_robust[chan], **kw)
    ax[0, 1].set_title(f"robust={r}")
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
    assert np.all(np.abs(img_uniform - img_robust) < 6e-3)

    plt.close("all")


def test_grid_natural(imager, tmp_path):
    kw = {"origin": "lower", "interpolation": "none", "extent": imager.coords.img_ext}

    chan = 0

    img_natural, beam_natural = imager.get_dirty_image(
        weighting="natural", check_visibility_scatter=False
    )

    r = 2
    img_robust, beam_robust = imager.get_dirty_image(
        weighting="briggs", robust=r, check_visibility_scatter=False
    )

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4.5))

    ax[0, 0].imshow(beam_natural[chan], **kw)
    ax[0, 0].set_title("natural")
    ax[1, 0].imshow(img_natural[chan], **kw)

    ax[0, 1].imshow(beam_robust[chan], **kw)
    ax[0, 1].set_title(f"robust={r}")
    ax[1, 1].imshow(img_robust[chan], **kw)

    # the differences
    im = ax[0, 2].imshow(beam_natural[chan] - beam_robust[chan], **kw)
    plt.colorbar(im, ax=ax[0, 2])
    ax[0, 2].set_title("difference")
    im = ax[1, 2].imshow(img_natural[chan] - img_robust[chan], **kw)
    plt.colorbar(im, ax=ax[1, 2])

    fig.subplots_adjust(left=0.05, right=0.95, wspace=0.02, bottom=0.07, top=0.94)

    fig.savefig(tmp_path / "grid_natural_v_robust.png", dpi=300)

    assert np.all(np.abs(beam_natural - beam_robust) < 1.5e-3)
    assert np.all(np.abs(img_natural - img_robust) < 3e-5)

    plt.close("all")


def test_grid_natural_arcsec2(imager, tmp_path):
    kw = {"origin": "lower", "interpolation": "none", "extent": imager.coords.img_ext}

    chan = 0

    img_natural, beam_natural = imager.get_dirty_image(
        weighting="natural", unit="Jy/arcsec^2", check_visibility_scatter=False
    )

    r = 2
    img_robust, beam_robust = imager.get_dirty_image(
        weighting="briggs", robust=r, unit="Jy/arcsec^2", check_visibility_scatter=False
    )

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 4.5))

    ax[0, 0].imshow(beam_natural[chan], **kw)
    ax[0, 0].set_title("natural")
    im = ax[1, 0].imshow(img_natural[chan], **kw)
    plt.colorbar(im, ax=ax[1, 0])

    ax[0, 1].imshow(beam_robust[chan], **kw)
    ax[0, 1].set_title(f"robust={r}")
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

    assert np.all(np.abs(beam_natural - beam_robust) < 1.5e-3)
    assert np.all(np.abs(img_natural - img_robust) <2e-4)

    plt.close("all")


def test_cell_variance_warning_image(mock_dataset_np):
    coords = coordinates.GridCoords(cell_size=0.01, npix=400)

    uu, vv, weight, data_re, data_im = mock_dataset_np
    sigma = np.sqrt(1 / weight)
    data_re = np.ones_like(uu) + np.random.normal(loc=0, scale=2 * sigma, size=uu.shape)
    data_im = np.zeros_like(uu) + np.random.normal(
        loc=0, scale=2 * sigma, size=uu.shape
    )

    imager = gridding.DirtyImager(
        coords=coords,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )

    with pytest.warns(RuntimeWarning):
        imager.get_dirty_image(weighting="uniform")
