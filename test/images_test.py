import pytest
import torch

from mpol import images


def test_odd_npix():
    with pytest.raises(AssertionError):
        images.ImageCube(npix=853, nchan=30, cell_size=0.015)


def test_negative_cell_size():
    with pytest.raises(AssertionError):
        images.ImageCube(npix=800, nchan=30, cell_size=-0.015)


def test_instantiate_image():
    images.ImageCube(npix=800, nchan=40, cell_size=0.015)


def test_instantiate_with_cube():
    cube = torch.full((800, 800, 40), 0.05, dtype=torch.double)
    images.ImageCube(npix=800, nchan=40, cell_size=0.015, cube=cube)


def test_pixel_mapping():
    images.ImageCube(
        npix=800, nchan=40, cell_size=0.015, pixel_mapping=torch.nn.Softplus(beta=10)
    )
