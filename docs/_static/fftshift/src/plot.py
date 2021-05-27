import argparse

parser = argparse.ArgumentParser(description="Create the fftshift plot")
parser.add_argument("outfile", help="Destination to save plot.")
args = parser.parse_args()

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file
from matplotlib import patches
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

from mpol import coordinates

fname = download_file(
    "https://zenodo.org/record/4711811/files/logo_cont.fits",
    cache=True,
    show_progress=True,
    pkgname="mpol",
)
coords = coordinates.GridCoords(cell_size=0.007, npix=512)
kw = {"origin": "lower", "interpolation": "none", "extent": coords.img_ext}
kwvis = {"origin": "lower", "interpolation": "none", "extent": coords.vis_ext}

fig = plt.figure(constrained_layout=False, figsize=(8, 6))
gs = GridSpec(2, 3, fig)
ax = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 2])


d = fits.open(fname)
sky_cube = d[0].data
d.close()
ax.imshow(sky_cube, **kw)
ax.set_title("Sky Cube")
ax.set_xlabel(r"$\Delta \alpha \cos \delta \; [{}^{\prime\prime}]$")
ax.set_ylabel(r"$\Delta \delta\; [{}^{\prime\prime}]$")

flip_cube = np.flip(sky_cube, (1,))
ax1.imshow(sky_cube, **kw)
ax1.set_title("Flip Cube")
ax1.set_xlabel(r"$\Delta \alpha \cos \delta\; [{}^{\prime\prime}]$")
ax1.set_ylabel(r"$\Delta \delta\; [{}^{\prime\prime}]$")
ax1.invert_xaxis()

image_cube = np.fft.fftshift(flip_cube, axes=(0, 1))
ax2.imshow(image_cube, **kw)
ax2.set_title("Packed Cube (Image)")
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)

visibility_cube = np.fft.fft2(image_cube, axes=(0, 1))
ax3.imshow(np.abs(visibility_cube), norm=LogNorm(vmin=1e-4, vmax=12000), **kwvis)
ax3.set_title("Packed Cube (Visibility)")
ax3.xaxis.set_visible(False)
ax3.yaxis.set_visible(False)

ground_cube = np.fft.fftshift(visibility_cube, axes=(0, 1))
ax4.imshow(np.abs(ground_cube), **kwvis, norm=LogNorm(vmin=1e-4, vmax=12000))
ax4.set_title("Ground Cube")
ax4.set_xlabel(r"$u$ [k$\lambda$]")
ax4.set_ylabel(r"$v$ [k$\lambda$]")

arrow_kws = {"mutation_scale": 20, "transform": fig.transFigure, "fc": "black"}

annotate_kws = {
    "xycoords": fig.transFigure,
    "va": "center",
    "ha": "center",
    "weight": "bold",
    "fontsize": "large",
}

text_kws = {"va": "center", "ha": "center", "weight": "bold", "fontsize": "large"}

y = 0.82
x0, x1 = 0.28, 0.37
arrow_sky_cube_to_flip_cube = patches.FancyArrowPatch(
    (x0, y), (x1, y), **arrow_kws, arrowstyle="<->"
)
fig.patches.append(arrow_sky_cube_to_flip_cube)
fig.text((x0 + x1) / 2, y + 0.05, "flip \n across R.A.", **text_kws)

x0, x1 = 0.62, 0.72
y = 0.84
arrow_flip_cube_to_packed_cube = patches.FancyArrowPatch(
    (x0, y), (x1, y), **arrow_kws, arrowstyle="->"
)
fig.patches.append(arrow_flip_cube_to_packed_cube)
fig.text((x0 + x1) / 2, y + 0.03, "fftshift", **text_kws)

x0, x1 = 0.72, 0.62
y = 0.77
arrow_packed_cube_to_flip_cube = patches.FancyArrowPatch(
    (x0, y), (x1, y), **arrow_kws, arrowstyle="->"
)
fig.patches.append(arrow_packed_cube_to_flip_cube)
fig.text((x0 + x1) / 2, y + 0.03, "ifftshift", **text_kws)


x_center = 0.5
arrow_packed_image_to_packed_visibility = patches.FancyArrowPatch(
    (0.73, 0.59), (0.32, 0.38), **arrow_kws, arrowstyle="->"
)
fig.patches.append(arrow_packed_image_to_packed_visibility)
fig.text(x_center, 0.50, "fft2", rotation=17, **text_kws)

arrow_packed_visibility_to_packed_image = patches.FancyArrowPatch(
    (0.34, 0.33), (0.75, 0.54), **arrow_kws, arrowstyle="->"
)
fig.patches.append(arrow_packed_visibility_to_packed_image)
fig.text(x_center, 0.37, "ifft2", rotation=17, **text_kws)


x0, x1 = 0.62, 0.31
y = 0.23
arrow_ground_cube_to_packed_cube = patches.FancyArrowPatch(
    (x0, y), (x1, y), **arrow_kws, arrowstyle="->"
)
fig.patches.append(arrow_ground_cube_to_packed_cube)
plt.annotate("fftshift", ((x0 + x1) / 2, y + 0.02), **annotate_kws)

x0, x1 = 0.31, 0.62
y = 0.16
arrow_packed_cube_to_ground_cube = patches.FancyArrowPatch(
    (x0, y), (x1, y), **arrow_kws, arrowstyle="->"
)
fig.patches.append(arrow_packed_cube_to_ground_cube)
plt.annotate("ifftshift", ((x0 + x1) / 2, y + 0.02), **annotate_kws)


fig.subplots_adjust(wspace=0.65, left=0.06, right=0.94, top=0.97, bottom=0.05)
plt.savefig(args.outfile, dpi=300)
