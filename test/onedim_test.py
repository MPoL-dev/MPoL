import matplotlib.pyplot as plt
import numpy as np

from mpol.onedim import radialI, radialV
from mpol.plot import plot_image

def test_radialI(mock_1d_image_model, tmp_path):
    # obtain a 1d radial brightness profile I(r) from an image    
    
    r, i, i2d, _, _, geom, coords = mock_1d_image_model

    bins = np.linspace(0, 2.0, 100)
    rtest, itest = radialI(i2d, coords, geom, bins=bins)

    fig, ax = plt.subplots(ncols=2)

    plot_image(i2d, extent=coords.img_ext, ax=ax[0], clab='Jy / sr')

    ax[0].title('AS 209-like profile.\nGeometry: {:}'.format(geom))

    ax[1].plot(r, i, 'k', label='truth')
    ax[1].plot(rtest, itest, 'r.-', label='recovery')
    
    ax[1].set_xlabel('r [arcsec]')
    ax[1].set_ylabel('I [Jy / sr]')

    fig.savefig(tmp_path / "test_radialI.png", dpi=300)
    plt.close("all")

    expected = [
        6.40747314e+10, 4.01920507e+10, 1.44803534e+10, 2.94238627e+09, 
        1.28782935e+10, 2.68613199e+10, 2.26564596e+10, 1.81151845e+10, 
        1.52128965e+10, 1.05640352e+10, 1.33411204e+10, 1.61124502e+10,
        1.41500539e+10, 1.20121195e+10, 1.11770326e+10, 1.19676913e+10, 
        1.20941686e+10, 1.09498286e+10, 9.74236410e+09, 7.99589196e+09, 
        5.94787809e+09, 3.82074946e+09, 1.80823933e+09, 4.48414819e+08,
        3.17808840e+08, 5.77317876e+08, 3.98851281e+08, 8.06459834e+08, 
        2.88706161e+09, 6.09577814e+09, 6.98556762e+09, 4.47436415e+09, 
        1.89511273e+09, 5.96604356e+08, 3.44571640e+08, 5.65906765e+08,
        2.85854589e+08, 2.67589013e+08, 3.98357054e+08, 2.97052261e+08, 
        3.82744591e+08, 3.52239791e+08, 2.74336969e+08, 2.28425747e+08, 
        1.82290043e+08, 3.16077299e+08, 1.18465538e+09, 3.32239287e+09,
        5.26718846e+09, 5.16458748e+09, 3.58114198e+09, 2.13431954e+09, 
        1.40936556e+09, 1.04032244e+09, 9.24050422e+08, 8.46829316e+08, 
        6.80909295e+08, 6.83812465e+08, 6.91856237e+08, 5.29227136e+08,
        3.97557293e+08, 3.54893419e+08, 2.60997039e+08, 2.09306498e+08, 
        1.93930693e+08, 6.97032407e+07, 6.66090083e+07, 1.40079594e+08, 
        7.21775931e+07, 3.23902663e+07, 3.35932300e+07, 7.63318789e+06,
        1.29740981e+07, 1.44300351e+07, 8.06249624e+06, 5.85567843e+06, 
        1.42637174e+06, 3.21445075e+06, 1.83763663e+06, 1.16926652e+07, 
        2.46918188e+07, 1.60206523e+07, 3.26596592e+06, 1.27837319e+05,
        2.27104612e+04, 4.77267063e+03, 2.90467640e+03, 2.88482230e+03, 
        1.43402521e+03, 1.54791996e+03, 7.23397046e+02, 1.02561351e+03, 
        5.24845888e+02, 1.47320552e+03, 7.40419174e+02, 4.59029378e-03,
        0.00000000e+00, 0.00000000e+00, 0.00000000e+00
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
