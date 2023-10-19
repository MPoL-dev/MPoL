import matplotlib.pyplot as plt
import numpy as np

from mpol.coordinates import GridCoords
from mpol.onedim import radialI, radialV
from mpol.precomposed import SimpleNet
from mpol.utils import sky_gaussian_arcsec

def test_radialV(coords, imager, dataset, generic_parameters):
    # obtain a 1d radial visibility profile V(q) from 2d visibilities
    coords = GridCoords(cell_size=0.005, npix=800)

    g = sky_gaussian_arcsec(coords.sky_x_centers_2D, coords.sky_y_centers_2D)
    
    nchan = dataset.nchan
    model = SimpleNet(coords=coords, nchan=nchan)

def test_radialI(coords, imager, dataset, generic_parameters):
    # obtain a 1d radial brightness profile I(r) from an image
    nchan = dataset.nchan
    model = SimpleNet(coords=coords, nchan=nchan)
