import pytest
from mpol import gridding
import numpy as np
import matplotlib.pyplot as plt


def test_grid_coords_instantiate():
    mycoords = gridding.GridCoords(cell_size=0.01, npix=512)


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
    data_im = d["data_im"]

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
    data_im = d["data_im"]

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
    data_im = d["data_im"]

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
    data_im = d["data_im"]

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


def test_gridder_conjugated(mock_visibility_data):
    coords = gridding.GridCoords(cell_size=0.005, npix=800)

    d = mock_visibility_data
    uu = d["uu"]
    vv = d["vv"]
    weight = 0.1 * np.ones_like(uu)
    data_re = np.ones_like(uu)
    data_im = np.zeros_like(uu)

    nchan, nvis_start = uu.shape

    gridder = gridding.Gridder(
        coords=coords, uu=uu, vv=vv, weight=weight, data_re=data_re, data_im=data_im,
    )

    nchan, nvis_end = gridder.uu.shape
    print("nvis start:{:} end:{:}".format(nvis_start, nvis_end))
    assert nvis_end / nvis_start == 2

    assert np.min(gridder.uu) > 0
    assert np.min(gridder.vv) < 0
    assert np.max(gridder.vv) > 0


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
    plt.savefig(tmp_path / "gridded_re.png", dpi=300)


# now that we've tested the creation ops, cache an instantiated gridder for future ops
@pytest.fixture
def gridder(mock_visibility_data):
    d = mock_visibility_data

    uu = d["uu"]
    vv = d["vv"]
    weight = d["weight"]
    data_re = d["data_re"]
    data_im = d["data_im"]

    return gridding.Gridder(
        cell_size=0.005,
        npix=800,
        uu=uu,
        vv=vv,
        weight=weight,
        data_re=data_re,
        data_im=data_im,
    )


# actually do the gridding
def test_grid_uniform(gridder):
    gridder.grid_visibilities()

    beam = gridder.get_dirty_beam()


# def test_grid_natural(gridder):
#     gridder.grid_visibilities(weighting="natural")


# def test_grid_briggs(gridder):
#     gridder.grid_visibilities(weighting="briggs", robust=0.5)


# make a dirty beam

# make a dirty image

# see if we can get it in the correct units
# export them
# does the imager work for single-channeled visiblities?

