# test the numpy matrices for interpolation

import pytest

import numpy as np
import matplotlib.pyplot as plt

from mpol import gridding

# convert from arcseconds to radians
arcsec = np.pi / (180.0 * 3600)  # [radians]  = 1/206265 radian/arcsec


def sky_plane(
    alpha,
    dec,
    a=1,
    delta_alpha=1.0 * arcsec,
    delta_delta=1.0 * arcsec,
    sigma_alpha=1.0 * arcsec,
    sigma_delta=1.0 * arcsec,
    Omega=0.0,
):
    """
    Calculates a Gaussian on the sky plane 

    Args:
        alpha: ra (in radians)
        delta: dec (in radians)
        a : amplitude
        delta_alpha : offset (in radians)
        delta_dec : offset (in radians)
        sigma_alpha : width (in radians)
        sigma_dec : width (in radians)
        Omega : position angle of ascending node (in degrees east of north)

    Returns:
        Gaussian evaluated at input args
    """

    return a * np.exp(
        -(
            (alpha - delta_alpha) ** 2 / (2 * sigma_alpha ** 2)
            + (dec - delta_delta) ** 2 / (2 * sigma_delta ** 2)
        )
    )


def fourier_plane(
    u,
    v,
    a=1,
    delta_alpha=1.0 * arcsec,
    delta_delta=1.0 * arcsec,
    sigma_alpha=1.0 * arcsec,
    sigma_delta=1.0 * arcsec,
    Omega=0.0,
):
    """
    Calculates the Analytic Fourier transform of the sky plane Gaussian. 

    Args:
        u: spatial freq (in kλ)
        v: spatial freq (in  kλ)
        a : amplitude
        delta_alpha : offset (in radians)
        delta_dec : offset (in radians)
        sigma_alpha : width (in radians)
        sigma_dec : width (in radians)
        Omega : position angle of ascending node (in degrees east of north)
    
    Returns:
        FT Gaussian evaluated at u, v points
    """

    # convert back to radians
    u = u * 1e3
    v = v * 1e3

    return (
        2
        * np.pi
        * a
        * sigma_alpha
        * sigma_delta
        * np.exp(
            -2 * np.pi ** 2 * (sigma_alpha ** 2 * u ** 2 + sigma_delta ** 2 * v ** 2)
            - 2 * np.pi * 1.0j * (delta_alpha * u + delta_delta * v)
        )
    )


# Let's plot this up and see what it looks like
N_alpha = 128
N_dec = 128
img_radius = 15.0 * arcsec

# full span of the image
ra = gridding.fftspace(img_radius, N_alpha)  # [arcsec]
dec = gridding.fftspace(img_radius, N_dec)  # [arcsec]

# fill out an image
img = np.empty((N_dec, N_alpha), np.float)

for i, delta in enumerate(dec):
    for j, alpha in enumerate(ra):
        img[i, j] = sky_plane(alpha, delta)

fig, ax = plt.subplots(nrows=1)
ax.imshow(img, origin="upper", interpolation="none", aspect="equal")
ax.set_xlabel(r"$\Delta \alpha \cos \delta$")
ax.set_ylabel(r"$\Delta \delta$")
ax.set_title("Input image")
fig.savefig("test/input.png", dpi=300)

# pre-multiply the image by the correction function
corrfun_mat = gridding.corrfun_mat(np.fft.fftshift(ra), np.fft.fftshift(dec))

fig, ax = plt.subplots(nrows=1)
im = ax.imshow(corrfun_mat, origin="upper", interpolation="none", aspect="equal")
plt.colorbar(im)
ax.set_xlabel(r"$\Delta \alpha \cos \delta$")
ax.set_ylabel(r"$\Delta \delta$")
ax.set_title("Correction function")
fig.savefig("test/corrfun.png", dpi=300)


# calculate the Fourier coordinates
dalpha = (2 * img_radius) / N_alpha
ddelta = (2 * img_radius) / N_dec

us = np.fft.rfftfreq(N_alpha, d=dalpha) * 1e-3  # convert to [kλ]
vs = np.fft.fftfreq(N_dec, d=ddelta) * 1e-3  # convert to [kλ]


# calculate the FFT, but first shift all axes.
# normalize output properly
vis = dalpha * ddelta * np.fft.rfftn(corrfun_mat * np.fft.fftshift(img), axes=(0, 1))
vis_no_cor = dalpha * ddelta * np.fft.rfftn(np.fft.fftshift(img), axes=(0, 1))

# calculate the corresponding u and v axes
XX, YY = np.meshgrid(us, vs)

# left, right, bottom, top
vs_limit = np.fft.fftshift(vs)

XX_full, YY_full = np.meshgrid(vs, vs)
vis_analytical_full = fourier_plane(XX_full, YY_full)

ext_full = [vs_limit[-1], vs_limit[0], vs_limit[-1], vs_limit[0]]
fig, ax = plt.subplots(nrows=2)
ax[0].imshow(
    np.real(np.fft.fftshift(vis_analytical_full)), origin="upper", extent=ext_full
)
ax[1].imshow(
    np.imag(np.fft.fftshift(vis_analytical_full)), origin="upper", extent=ext_full
)
fig.savefig("test/analytical_full.png", dpi=300)


ext = [us[0], us[-1], vs_limit[-1], vs_limit[0]]

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(7, 5))
ax[0, 0].set_title("numerical")
ax[0, 0].imshow(
    np.real(np.fft.fftshift(vis_no_cor, axes=0)),
    origin="upper",
    interpolation="none",
    aspect="equal",
    extent=ext,
)
ax[1, 0].imshow(
    np.imag(np.fft.fftshift(vis_no_cor, axes=0)),
    origin="upper",
    interpolation="none",
    aspect="equal",
    extent=ext,
)

ax[0, 1].set_title("analytical")
vis_analytical = fourier_plane(XX, YY)
im_ral = ax[0, 1].imshow(
    np.real(np.fft.fftshift(vis_analytical, axes=0)),
    origin="upper",
    interpolation="none",
    aspect="equal",
    extent=ext,
)
plt.colorbar(im_ral, ax=ax[0, 1])
ax[1, 1].imshow(
    np.imag(np.fft.fftshift(vis_analytical, axes=0)),
    origin="upper",
    interpolation="none",
    aspect="equal",
    extent=ext,
)

# compare to the analytic version
vis_diff = np.fft.fftshift(vis_no_cor - vis_analytical, axes=0)

np.imag(np.fft.fftshift(vis_no_cor - vis_analytical, axes=0))
ax[0, 2].set_title("difference")
im_real = ax[0, 2].imshow(
    np.real(vis_diff), origin="upper", interpolation="none", aspect="equal", extent=ext
)
plt.colorbar(im_real, ax=ax[0, 2])
im_imag = ax[1, 2].imshow(
    np.imag(vis_diff), origin="upper", interpolation="none", aspect="equal", extent=ext
)
plt.colorbar(im_imag, ax=ax[1, 2])

fig.savefig("test/output.png", dpi=300, wspace=0.05)


def test_2D_interpolation():
    # create a numerical test to make sure the difference is small
    assert np.sum(np.abs(vis_diff)) < 1e-10, "visibilities don't match"


# create a dataset with baselines
np.random.seed(42)
N_vis = 50
data_points = np.random.uniform(
    low=0.9 * np.min(vs), high=0.9 * np.max(vs), size=(N_vis, 2)
)

# rather than completely random, let's try to choose something that sort of makes sense to compare to a real array


# fig, ax = plt.subplots(nrows=1)
# ax.scatter(u_data, v_data)
# fig.savefig("baselines.png", dpi=300)

# First let's test the intpolation on some known u, v points that will have large visibility amplitudes
# and some that suffer from edge cases
# from the figures, these could be
data_points = np.array(
    [
        [50.0, 10.0],
        [50.0, 0.0],
        [50.0, -1.0],
        [-50.0, 10.0],
        [5.0, 1.0],
        [-5.0, 1.0],
        [5.0, 20.0],
        [-5.0, -20.0],
    ]
)

# data_points = np.array([[5.0, 1.0], [-5.0, 1.0]])

u_data, v_data = data_points.T

data_values = fourier_plane(u_data, v_data)

# calculate and visualize the C_real and C_imag matrices
# these are scipy csc sparse matrices
C_real, C_imag = gridding.calc_matrices(u_data, v_data, us, vs)

fig, ax = plt.subplots(nrows=1, figsize=(6, 6))
vvmax = np.max(np.abs(C_real.toarray()[:, 0:300]))
ax.imshow(
    C_real.toarray()[:, 0:300],
    interpolation="none",
    origin="upper",
    cmap="RdBu",
    aspect="auto",
    vmin=-vvmax,
    vmax=vvmax,
)
# ax[1].spy(C_real[:, 0:300], marker=".", precision="present", aspect="auto")
fig.savefig("test/C_real.png", dpi=300)
#
# fig, ax = plt.subplots(ncols=1, figsize=(12,3))
# vvmax = np.max(np.abs(C_imag[:,0:300]))
# ax.imshow(C_imag[:,0:300], interpolation="none", origin="upper", cmap="RdBu", aspect="auto", vmin=-vvmax, vmax=vvmax)
# fig.savefig("C_imag.png", dpi=300)

# interpolated points
interp_real = C_real.dot(np.real(vis.flatten()))
interp_imag = C_imag.dot(np.imag(vis.flatten()))

fig, ax = plt.subplots(nrows=4, figsize=(4, 5))
ax[0].plot(np.real(data_values), ".", ms=4)
ax[0].plot(interp_real, ".", ms=3)
ax[0].set_ylabel("real")
ax[1].plot(interp_real - np.real(data_values), ".")
ax[1].set_ylabel("real diff")
ax[2].plot(np.imag(data_values), ".", ms=4)
ax[2].plot(interp_imag, ".", ms=3)
ax[2].set_ylabel("imag")
ax[3].plot(interp_imag - np.imag(data_values), ".")
ax[3].set_ylabel("imag diff")
fig.subplots_adjust(hspace=0.4, left=0.2)
fig.savefig("test/real_comp.png", dpi=300)
