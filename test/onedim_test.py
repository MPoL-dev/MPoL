import matplotlib.pyplot as plt
import numpy as np

from mpol.onedim import radialI, radialV
from mpol.plot import plot_image

def test_radialI(mock_1d_image_model):
    # obtain a 1d radial brightness profile I(r) from an image    
    
    r, i, i2d, _, _, geom, coords = mock_1d_image_model

    rtest, itest = radialI(i2d, coords, geom, bins=None)

    _, ax = plt.subplots(ncols=2)

    plot_image(i2d, extent=coords.img_ext, ax=ax[0], clab='Jy sr$^{-2}$')

    ax[0].title('AS 209-like profile.\nGeometry: {:}'.format(geom))

    ax[1].plot(r, i, 'k', label='truth')
    ax[1].plot(rtest, itest, 'r.-', label='result')

    expected = [
        5.79780326e+10, 2.47990375e+10, 4.19794053e+09, 1.63165616e+10, 
        2.56197452e+10, 1.86014523e+10, 1.39800643e+10, 1.14935415e+10, 
        1.58898181e+10, 1.36344624e+10, 1.14059388e+10, 1.18389766e+10,
        1.18678220e+10, 1.02746571e+10, 8.18228608e+09, 5.40044021e+09, 
        2.63008657e+09, 5.61017562e+08, 4.11251459e+08, 4.85212055e+08, 
        7.86240201e+08, 3.80008818e+09, 6.99254078e+09, 4.63422518e+09,
        1.47955225e+09, 3.54437101e+08, 5.40245124e+08, 2.40733707e+08, 
        3.75558288e+08, 3.09550836e+08, 3.89050755e+08, 2.94034065e+08, 
        2.28420989e+08, 2.02024946e+08, 8.01676870e+08, 3.32150158e+09,
        5.39650629e+09, 4.14278723e+09, 2.15660813e+09, 1.26767762e+09, 
        9.44844287e+08, 8.38469945e+08, 6.67265501e+08, 7.01335008e+08, 
        5.23378469e+08, 3.81449784e+08, 2.90773229e+08, 2.11627913e+08,
        1.49714706e+08, 5.25051148e+07, 1.32583044e+08, 4.70658247e+07, 
        3.52859146e+07, 8.45017542e+06, 1.55314270e+07, 9.15833896e+06, 
        5.27578119e+06, 2.08955802e+06, 2.20079612e+06, 1.36049060e+07,
        2.26196295e+07, 5.66055107e+06, 1.30125309e+05, 1.23963592e+04, 
        3.63853801e+03, 2.74032116e+03, 1.21440814e+03, 1.00115299e+03, 
        9.39440210e+02, 8.74909523e+02, 1.01296747e+03, 3.19296404e-02,
        0.00000000e+00, 0.00000000e+00
        ]

    np.testing.assert_allclose(itest, expected, rtol=1e-6,
                               err_msg="test_radialI")


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
