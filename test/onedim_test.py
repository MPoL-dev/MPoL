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

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    plot_image(
        np.squeeze(torch2npy(icube.sky_cube)),
        extent=icube.coords.img_ext,
        ax=ax[0],
        clab="Jy / sr",
    )

    ax[1].plot(rtrue, itrue, "k", label="truth")
    ax[1].plot(rtest, itest, "r.-", label="recovery")

    ax[0].set_title(f"Geometry:\n{geom}", fontsize=7)

    ax[1].set_xlabel("r [arcsec]")
    ax[1].set_ylabel("I [Jy / sr]")
    ax[1].legend()

    fig.savefig(tmp_path / "test_radialI.png", dpi=300)
    plt.close("all")

    expected = [
        6.40747314e10,
        4.01920507e10,
        1.44803534e10,
        2.94238627e09,
        1.28782935e10,
        2.68613199e10,
        2.26564596e10,
        1.81151845e10,
        1.52128965e10,
        1.05640352e10,
        1.33411204e10,
        1.61124502e10,
        1.41500539e10,
        1.20121195e10,
        1.11770326e10,
        1.19676913e10,
        1.20941686e10,
        1.09498286e10,
        9.74236410e09,
        7.99589196e09,
        5.94787809e09,
        3.82074946e09,
        1.80823933e09,
        4.48414819e08,
        3.17808840e08,
        5.77317876e08,
        3.98851281e08,
        8.06459834e08,
        2.88706161e09,
        6.09577814e09,
        6.98556762e09,
        4.47436415e09,
        1.89511273e09,
        5.96604356e08,
        3.44571640e08,
        5.65906765e08,
        2.85854589e08,
        2.67589013e08,
        3.98357054e08,
        2.97052261e08,
        3.82744591e08,
        3.52239791e08,
        2.74336969e08,
        2.28425747e08,
        1.82290043e08,
        3.16077299e08,
        1.18465538e09,
        3.32239287e09,
        5.26718846e09,
        5.16458748e09,
        3.58114198e09,
        2.13431954e09,
        1.40936556e09,
        1.04032244e09,
        9.24050422e08,
        8.46829316e08,
        6.80909295e08,
        6.83812465e08,
        6.91856237e08,
        5.29227136e08,
        3.97557293e08,
        3.54893419e08,
        2.60997039e08,
        2.09306498e08,
        1.93930693e08,
        6.97032407e07,
        6.66090083e07,
        1.40079594e08,
        7.21775931e07,
        3.23902663e07,
        3.35932300e07,
        7.63318789e06,
        1.29740981e07,
        1.44300351e07,
        8.06249624e06,
        5.85567843e06,
        1.42637174e06,
        3.21445075e06,
        1.83763663e06,
        1.16926652e07,
        2.46918188e07,
        1.60206523e07,
        3.26596592e06,
        1.27837319e05,
        2.27104612e04,
        4.77267063e03,
        2.90467640e03,
        2.88482230e03,
        1.43402521e03,
        1.54791996e03,
        7.23397046e02,
        1.02561351e03,
        5.24845888e02,
        1.47320552e03,
        7.40419174e02,
        4.59029378e-03,
        0.00000000e00,
        0.00000000e00,
        0.00000000e00,
    ]

    np.testing.assert_allclose(itest, expected, rtol=1e-6, err_msg="test_radialI")


def test_radialV(mock_1d_vis_model, tmp_path):
    # obtain a 1d radial visibility profile V(q) from 2d visibilities

    fcube, Vtrue_dep, q_dep, geom = mock_1d_vis_model

    bins = np.linspace(1, 5e3, 100)

    qtest, Vtest = radialV(fcube, geom, rescale_flux=True, bins=bins)

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(10, 10))

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
        -9.61751019e09,
        2.75229026e09,
        -4.36137738e08,
        -2.30171445e07,
        -2.10099938e08,
        2.86360366e08,
        -1.37544187e07,
        -3.62764471e07,
        1.94332782e07,
        -4.63579878e07,
        4.38157379e07,
        -1.19891002e07,
        2.47285137e07,
        -3.43389203e07,
        7.49974578e05,
        3.68423107e06,
        9.43443498e06,
        -1.16182426e07,
        1.08867793e07,
        -8.74943322e06,
        1.14521810e07,
        -6.36361380e06,
        3.58538842e05,
        -5.96714707e06,
        1.04348614e07,
        -1.47220982e06,
        -1.19522309e07,
        -4.09776593e06,
        7.86540505e06,
        3.60337006e06,
        -8.30025685e06,
        4.05093017e06,
        3.33292357e06,
        2.05733741e06,
        -7.65245396e06,
        3.73332165e06,
        3.40645897e06,
        -4.58494946e06,
        3.66101584e06,
        -3.69627118e06,
        5.27955178e06,
        9.75812262e06,
        -1.65425072e07,
        5.47225658e06,
        -3.49680316e06,
        8.22030443e06,
        -7.32448474e06,
        -4.23843848e06,
        1.27346507e07,
        -4.60792496e06,
        -2.56148856e06,
        6.29770245e05,
        -2.25521550e06,
        5.35018477e06,
        -4.61334469e06,
        3.09166148e06,
        -9.18155255e05,
        -1.00736465e06,
        1.12177040e06,
        -9.21570359e05,
        8.70817075e05,
        3.16472432e04,
        -1.59681139e06,
        1.16213263e06,
        3.64004059e04,
        -8.49130119e04,
        -2.30599556e05,
        -1.59965392e03,
        9.30837779e05,
        -3.90387012e05,
        -4.75338516e05,
        7.53183050e04,
        3.41897054e05,
        -7.53936979e05,
        1.99039974e06,
        -1.90488504e06,
        -4.19283666e05,
        1.53004765e06,
        -8.55774990e05,
        6.21661335e05,
        -5.00689314e05,
        -6.26249184e05,
        1.94062725e06,
        -1.65756778e06,
        1.03046094e06,
        -7.77307547e05,
        1.65177536e05,
        1.07726803e05,
        -2.46681205e05,
        1.18707317e05,
        7.05201176e04,
        1.39152470e05,
        -2.80631868e04,
        4.92257795e05,
        -1.52044894e06,
        1.02459630e06,
        8.42494484e05,
        -1.57362080e06,
        7.22603120e05,
    ]

    np.testing.assert_allclose(Vtest.real, expected, rtol=1e-6, err_msg="test_radialV")
