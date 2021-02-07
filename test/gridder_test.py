import pytest
from mpol import gridding
import numpy as np
import matplotlib.pyplot as plt


def test_grid_coords_instantiate():
    coords = gridding.GridCoords(cell_size=0.01, npix=512)


def test_grid_coords_equal():
    coords1 = gridding.GridCoords(cell_size=0.01, npix=512)
    coords2 = gridding.GridCoords(cell_size=0.01, npix=512)

    assert coords1 == coords2


def test_grid_coords_unequal_pix():
    coords1 = gridding.GridCoords(cell_size=0.01, npix=510)
    coords2 = gridding.GridCoords(cell_size=0.01, npix=512)

    assert coords1 != coords2


def test_grid_coords_unequal_cell_size():
    coords1 = gridding.GridCoords(cell_size=0.011, npix=512)
    coords2 = gridding.GridCoords(cell_size=0.01, npix=512)

    assert coords1 != coords2


def test_grid_coords_odd_fail():
    with pytest.raises(AssertionError):
        mycoords = gridding.GridCoords(cell_size=0.01, npix=511)


def test_grid_coords_neg_cell_size():
    with pytest.raises(AssertionError):
        mycoords = gridding.GridCoords(cell_size=-0.01, npix=512)


# instantiate a Gridder object with mock visibilities
def test_grid_coords_fit(mock_visibility_data):
    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]

    mycoords = gridding.GridCoords(cell_size=0.005, npix=800)
    mycoords.check_data_fit(uu, vv)


def test_grid_coords_fail(mock_visibility_data):
    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]

    mycoords = gridding.GridCoords(cell_size=0.05, npix=800)

    print("max u data", np.max(uu))
    print("max u grid", mycoords.max_grid)

    with pytest.raises(AssertionError):
        mycoords.check_data_fit(uu, vv)


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

    mycoords = gridding.GridCoords(cell_size=0.005, npix=800)

    gridding.Gridder(
        coords=mycoords, uu=uu, vv=vv, weight=weight, data_re=data_re, data_im=data_im,
    )


def test_gridder_instantiate_npix_gridCoord_conflict(mock_visibility_data):
    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data_re = d["data_re"]
    data_im = -d["data_im"]

    mycoords = gridding.GridCoords(cell_size=0.005, npix=800)

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

    mycoords = gridding.GridCoords(cell_size=0.05, npix=800)

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
    coords = gridding.GridCoords(cell_size=0.005, npix=800)

    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]
    weight = 0.1 * np.ones_like(uu)
    data_re = np.ones_like(uu)
    data_im = np.zeros_like(uu)

    gridder = gridding.Gridder(
        coords=coords, uu=uu, vv=vv, weight=weight, data_re=data_re, data_im=data_im,
    )

    # with uniform weighting, the gridded sheet should be uniform and = 1
    gridder.grid_visibilities(weighting="uniform")

    print(
        "re",
        np.mean(gridder.gridded_re),
        np.std(gridder.gridded_re),
        np.min(gridder.gridded_re),
        np.max(gridder.gridded_re),
    )

    assert pytest.approx(np.min(gridder.gridded_re), 0)
    assert pytest.approx(np.max(gridder.gridded_re), 1)

    im = plt.imshow(
        gridder.gridded_re[4], origin="lower", extent=gridder.coords.vis_ext
    )
    plt.colorbar(im)
    plt.savefig(str(tmp_path / "gridded_re.png"), dpi=300)


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
            gridder.grid_visibilities(weighting=weighting, robust=r)
        else:
            gridder.grid_visibilities(weighting=weighting)
        beam = gridder.get_dirty_beam()

        for i in range(gridder.nchan):
            assert pytest.approx(np.max(beam[i]), 1.0)


# compare uniform and robust = -2.0
def test_grid_uniform(gridder, tmp_path):

    kw = {"origin": "lower", "interpolation": "none", "extent": gridder.coords.img_ext}

    chan = 4

    gridder.grid_visibilities(weighting="uniform")
    beam_uniform = gridder.get_dirty_beam()
    img_uniform = gridder.get_dirty_image()

    r = -2
    gridder.grid_visibilities(weighting="briggs", robust=r)
    beam_robust = gridder.get_dirty_beam()
    img_robust = gridder.get_dirty_image()

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

    fig.savefig(str(tmp_path / "uniform_v_robust.png"), dpi=300)

    assert np.all(np.abs(beam_uniform - beam_robust) < 1e-4)
    assert np.all(np.abs(img_uniform - img_robust) < 1e-5)


def test_grid_natural(gridder, tmp_path):

    kw = {"origin": "lower", "interpolation": "none", "extent": gridder.coords.img_ext}

    chan = 4

    gridder.grid_visibilities(weighting="natural")
    beam_natural = gridder.get_dirty_beam()
    img_natural = gridder.get_dirty_image()

    r = 2
    gridder.grid_visibilities(weighting="briggs", robust=r)
    beam_robust = gridder.get_dirty_beam()
    img_robust = gridder.get_dirty_image()

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

    fig.savefig(str(tmp_path / "natural_v_robust.png"), dpi=300)

    assert np.all(np.abs(beam_natural - beam_robust) < 1e-3)
    assert np.all(np.abs(img_natural - img_robust) < 1e-3)

