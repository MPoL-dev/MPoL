import matplotlib.pyplot as plt
import numpy as np
from mpol.onedim import radialI, radialV
from mpol.plot import plot_image
from mpol.utils import torch2npy


def test_radialI(mock_1d_image_model, tmp_path):
    # obtain a 1d radial brightness profile I(r) from an image

    rtrue, itrue, icube, _, _, geom = mock_1d_image_model

    bins = np.linspace(0, 2.0, 100)

    rtest, itest = radialI(icube, geom, bins=bins)

    fig, ax = plt.subplots(ncols=2, figsize=(10,5))

    plot_image(np.squeeze(torch2npy(icube.sky_cube)), extent=icube.coords.img_ext,
               ax=ax[0], clab="Jy / sr")

    ax[1].plot(rtrue, itrue, "k", label="truth")
    ax[1].plot(rtest, itest, "r.-", label="recovery")

    ax[0].set_title(f"Geometry:\n{geom}", fontsize=7)

    ax[1].set_xlabel("r [arcsec]")
    ax[1].set_ylabel("I [Jy / sr]")
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

    qtest, Vtest = radialV(fcube, geom, rescale_flux=True, bins=bins)

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(10,10))

    ax[0].plot(q_dep / 1e6, Vtrue_dep.real, "k.", label="truth deprojected")
    ax[0].plot(qtest / 1e3, Vtest.real, "r.-", label="recovery")

    ax[1].plot(q_dep / 1e6, Vtrue_dep.imag, "k.")
    ax[1].plot(qtest / 1e3, Vtest.imag, "r.")

    ax[0].set_xlim(-0.5, 6)
    ax[1].set_xlim(-0.5, 6)
    ax[1].set_xlabel(r"Baseline [M$\lambda$]")
    ax[0].set_ylabel("Re(V) [Jy]")
    ax[1].set_ylabel("Im(V) [Jy]")
    ax[0].set_title(f"Geometry {geom}", fontsize=10)
    ax[0].legend()

    fig.savefig(tmp_path / "test_radialV.png", dpi=300)
    plt.close("all")

    expected = [
        -9.61751019e+09,  2.75229026e+09, -4.36137738e+08, -2.30171445e+07,
        -2.10099938e+08,  2.86360366e+08, -1.37544187e+07, -3.62764471e+07,
        1.94332782e+07, -4.63579878e+07,  4.38157379e+07, -1.19891002e+07,
        2.47285137e+07, -3.43389203e+07,  7.49974578e+05,  3.68423107e+06,
        9.43443498e+06, -1.16182426e+07,  1.08867793e+07, -8.74943322e+06,
        1.14521810e+07, -6.36361380e+06,  3.58538842e+05, -5.96714707e+06,
        1.04348614e+07, -1.47220982e+06, -1.19522309e+07, -4.09776593e+06,
        7.86540505e+06,  3.60337006e+06, -8.30025685e+06,  4.05093017e+06,
        3.33292357e+06,  2.05733741e+06, -7.65245396e+06,  3.73332165e+06,
        3.40645897e+06, -4.58494946e+06,  3.66101584e+06, -3.69627118e+06,
        5.27955178e+06,  9.75812262e+06, -1.65425072e+07,  5.47225658e+06,
        -3.49680316e+06,  8.22030443e+06, -7.32448474e+06, -4.23843848e+06,
        1.27346507e+07, -4.60792496e+06, -2.56148856e+06,  6.29770245e+05,
        -2.25521550e+06,  5.35018477e+06, -4.61334469e+06,  3.09166148e+06,
        -9.18155255e+05, -1.00736465e+06,  1.12177040e+06, -9.21570359e+05,
        8.70817075e+05,  3.16472432e+04, -1.59681139e+06,  1.16213263e+06,
        3.64004059e+04, -8.49130119e+04, -2.30599556e+05, -1.59965392e+03,
        9.30837779e+05, -3.90387012e+05, -4.75338516e+05,  7.53183050e+04,
        3.41897054e+05, -7.53936979e+05,  1.99039974e+06, -1.90488504e+06,
        -4.19283666e+05,  1.53004765e+06, -8.55774990e+05,  6.21661335e+05,
        -5.00689314e+05, -6.26249184e+05,  1.94062725e+06, -1.65756778e+06,
        1.03046094e+06, -7.77307547e+05,  1.65177536e+05,  1.07726803e+05,
        -2.46681205e+05,  1.18707317e+05,  7.05201176e+04,  1.39152470e+05,
        -2.80631868e+04,  4.92257795e+05, -1.52044894e+06,  1.02459630e+06,
        8.42494484e+05, -1.57362080e+06,  7.22603120e+05
        ]

    np.testing.assert_allclose(Vtest.real, expected, rtol=1e-6,
                               err_msg="test_radialV")
