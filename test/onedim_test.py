import matplotlib.pyplot as plt
import numpy as np

from mpol.onedim import radialI, radialV
from mpol.plot import plot_image

def test_radialI(mock_1d_image_model, tmp_path):
    # obtain a 1d radial brightness profile I(r) from an image    

    rtrue, itrue, icube, _, _, geom = mock_1d_image_model

    bins = np.linspace(0, 2.0, 100)

    rtest, itest = radialI(i2dtrue, coords, geom, bins=bins)

    fig, ax = plt.subplots(ncols=2, figsize=(10,5))

    plot_image(i2dtrue, extent=coords.img_ext, ax=ax[0], clab='Jy / sr')

    ax[1].plot(rtrue, itrue, 'k', label='truth')
    ax[1].plot(rtest, itest, 'r.-', label='recovery')
    
    ax[0].set_title(f"Geometry:\n{geom}", fontsize=7)
    
    ax[1].set_xlabel('r [arcsec]')
    ax[1].set_ylabel('I [Jy / sr]')
    ax[1].legend()

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


def test_radialV(mock_1d_vis_model, tmp_path):
    # obtain a 1d radial visibility profile V(q) from 2d visibilities

    fcube, Vtrue_dep, q_dep, geom = mock_1d_vis_model

    bins = np.linspace(1,5e3,100)

    qtest, Vtest = radialV(Vtrue, coords, geom, rescale_flux=True, bins=bins)

    fig, ax = plt.subplots()

    ax.plot(q_dep / 1e6, Vtrue_dep, 'k.', label='truth deprojected')
    ax.plot(qtest / 1e3, Vtest, 'r.-', label='recovery')

    ax.set_xlim(-0.5, 6)
    ax.set_xlabel(r'Baseline [M$\lambda$]')
    ax.set_ylabel('Re(V) [Jy]')
    ax.set_title(f"Geometry {geom}", fontsize=10)
    ax.legend()

    fig.savefig(tmp_path / "test_radialV.png", dpi=300)
    plt.close("all")

    expected = [
        2.53998336e-01,  1.59897580e-01,  8.59460326e-02,  7.42189236e-02,  
        5.75440687e-02,  1.81324892e-02, -2.92922689e-03,  3.14354163e-03,  
        6.72339399e-03, -8.54632390e-03, -1.73385166e-02, -4.03826092e-03,
        1.45595908e-02,  1.61681713e-02,  5.93475866e-03,  2.45555912e-04, 
        -7.05014619e-04, -6.09266430e-03, -1.02454088e-02, -2.80944776e-03,  
        8.58212558e-03, 8.39132158e-03, -6.52523293e-04,  -4.34778158e-03,
        1.08035980e-04,  3.40903070e-03,  2.26682041e-03,  2.42465437e-03,  
        5.07968926e-03,  4.83377443e-03, 1.26300648e-03,  1.11930639e-03,  
        6.45157513e-03, 1.05751150e-02,  9.14016956e-03,  5.92209210e-03,
        5.18455986e-03,  5.88802559e-03,  5.49315770e-03, 4.96398638e-03,  
        5.81115311e-03,  5.95304063e-03, 3.16208083e-03, -1.71765038e-04, 
        -4.64532628e-04, 1.12448670e-03,  1.84297313e-03,  1.48094594e-03,
        1.12953770e-03,  1.01370816e-03,  6.57047907e-04, 1.37570722e-04,  
        3.00648884e-04,  1.04847404e-03, 1.16856102e-03,  3.08940761e-04, 
        -5.65721897e-04, -8.38907531e-04, -8.71976125e-04, -1.09567680e-03,
        -1.42077854e-03, -1.33702627e-03, -9.96839047e-04, -1.16400192e-03, 
        -1.43584618e-03, -1.07454472e-03, -6.44900590e-04, -4.86165342e-04, 
        -1.96851463e-04, 5.04190986e-05,  5.73950179e-05,  2.79905736e-04,
        7.52685847e-04,  1.12546048e-03,  1.37149548e-03, 1.35835560e-03,  
        1.06470794e-03,  8.81423014e-04, 8.76827161e-04,  9.03579902e-04,  
        8.39818807e-04, 5.19936424e-04,  1.46415537e-04,  3.29054769e-05,
        7.30096312e-05,  6.47553400e-05,  2.18817382e-05, 4.47955432e-06,  
        7.34705616e-06,  9.06184045e-06, 9.45269846e-06,  1.00464939e-05,  
        8.28166011e-06, 7.09361681e-06,  6.43221021e-06,  3.12425880e-06,
        2.57495214e-07,  6.48560373e-07,  1.88421498e-07
        ]

    np.testing.assert_allclose(Vtest, expected, rtol=1e-6,
                               err_msg="test_radialV")    