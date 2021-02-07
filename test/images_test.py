import pytest
import torch

from mpol import images


def test_odd_npix():
    with pytest.raises(AssertionError):
        images.BaseCube(npix=853, nchan=30, cell_size=0.015)

    with pytest.raises(AssertionError):
        images.ImageCube(npix=853, nchan=30, cell_size=0.015)


def test_negative_cell_size():
    with pytest.raises(AssertionError):
        images.BaseCube(npix=800, nchan=30, cell_size=-0.015)

    with pytest.raises(AssertionError):
        images.ImageCube(npix=800, nchan=30, cell_size=-0.015)


def test_single_chan():
    im = images.ImageCube(cell_size=0.015, npix=800)
    assert im.nchan == 1


# test image packing
# create fixture with GridCoord
# evaluate test analytic function over some channels
# plot, make sure shifts appear correct
# also compare FFT output to analytic answer (unit scaling)

# test basecube pixel mapping
# using known input cube, known mapping function, compare output
