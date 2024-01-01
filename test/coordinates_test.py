import matplotlib.pyplot as plt
import torch
import pytest

from mpol import coordinates
from mpol.constants import *
from mpol.exceptions import CellSizeError


def test_grid_coords_instantiate():
    coordinates.GridCoords(cell_size=0.01, npix=512)


def test_grid_coords_equal():
    coords1 = coordinates.GridCoords(cell_size=0.01, npix=512)
    coords2 = coordinates.GridCoords(cell_size=0.01, npix=512)

    assert coords1 == coords2


def test_grid_coords_unequal_pix():
    coords1 = coordinates.GridCoords(cell_size=0.01, npix=510)
    coords2 = coordinates.GridCoords(cell_size=0.01, npix=512)

    assert coords1 != coords2


def test_grid_coords_unequal_cell_size():
    coords1 = coordinates.GridCoords(cell_size=0.011, npix=512)
    coords2 = coordinates.GridCoords(cell_size=0.01, npix=512)

    assert coords1 != coords2


def test_grid_coords_plot_2D_uvq_sky(tmp_path):
    coords = coordinates.GridCoords(cell_size=0.005, npix=800)

    ikw = {"origin": "lower"}

    fig, ax = plt.subplots(nrows=1, ncols=3)
    im = ax[0].imshow(coords.sky_u_centers_2D, **ikw)
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(coords.sky_v_centers_2D, **ikw)
    plt.colorbar(im, ax=ax[1])

    im = ax[2].imshow(coords.sky_q_centers_2D, **ikw)
    plt.colorbar(im, ax=ax[2])

    for a, t in zip(ax, ["u", "v", "q"]):
        a.set_title(t)

    fig.savefig(tmp_path / "sky_uvq.png", dpi=300)


def test_grid_coords_plot_2D_uvq_packed(tmp_path):
    coords = coordinates.GridCoords(cell_size=0.005, npix=800)

    ikw = {"origin": "lower"}

    fig, ax = plt.subplots(nrows=1, ncols=3)
    im = ax[0].imshow(coords.packed_u_centers_2D, **ikw)
    plt.colorbar(im, ax=ax[0])

    im = ax[1].imshow(coords.packed_v_centers_2D, **ikw)
    plt.colorbar(im, ax=ax[1])

    im = ax[2].imshow(coords.packed_q_centers_2D, **ikw)
    plt.colorbar(im, ax=ax[2])

    for a, t in zip(ax, ["u", "v", "q"]):
        a.set_title(t)

    fig.savefig(tmp_path / "packed_uvq.png", dpi=300)


def test_grid_coords_odd_fail():
    with pytest.raises(ValueError, match="Image must have an even number of pixels."):
        coordinates.GridCoords(cell_size=0.01, npix=511)


def test_grid_coords_neg_cell_size():
    with pytest.raises(ValueError, match="cell_size must be a positive real number."):
        coordinates.GridCoords(cell_size=-0.01, npix=512)


# instantiate a DataAverager object with mock visibilities
def test_grid_coords_fit(baselines_2D):
    uu, vv = baselines_2D

    coords = coordinates.GridCoords(cell_size=0.005, npix=800)
    coords.check_data_fit(uu, vv)


def test_grid_coords_fail(baselines_2D):
    uu, vv = baselines_2D

    coords = coordinates.GridCoords(cell_size=0.05, npix=800)

    print("max u data", torch.max(uu))
    print("max u grid", coords.max_grid)

    with pytest.raises(CellSizeError):
        coords.check_data_fit(uu, vv)


def test_coords_mock_image(coords, img2D_butterfly):
    npix, _ = img2D_butterfly.shape
    assert coords.npix == npix, "coords dimensions and mock image have different sizes"
