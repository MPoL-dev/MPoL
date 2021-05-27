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

fig = plt.figure(constrained_layout=False)
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
ax.set_xlabel(r"$\alpha$ [deg]")
ax.set_ylabel(r"$\delta$ [deg]")

flip_cube = np.flip(sky_cube, (1,))
ax1.imshow(sky_cube, **kw)
ax1.set_title("Flip Cube")
ax1.set_xlabel(r"$\alpha$ [deg]")
ax1.set_ylabel(r"$\delta$ [deg]")
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


arrow = patches.FancyArrowPatch(
    (0.7, 0.3),
    (0.32, 0.3),
    transform=fig.transFigure,
    arrowstyle="simple",
    mutation_scale=30,
    fc='black'
)
arrow1 = patches.FancyArrowPatch(
    (0.65, 0.75),
    (0.78, 0.75),
    transform=fig.transFigure,
    arrowstyle="simple",
    mutation_scale=30,
    fc='black'
)
arrow2 = patches.FancyArrowPatch(
    (0.29, 0.75),
    (0.41, 0.75),
    transform=fig.transFigure,
    arrowstyle="simple",
    mutation_scale=30,
    fc='black'
)
arrow3 = patches.FancyArrowPatch(
    (0.75, 0.57),
    (0.32, 0.36),
    transform=fig.transFigure,
    arrowstyle="simple",
    mutation_scale=30,
    fc='black'
)
arrow4 = patches.FancyArrowPatch(
    (0.32, 0.33),
    (0.75, 0.54),
    transform=fig.transFigure,
    arrowstyle="simple",
    mutation_scale=30,
    fc='black'
)

fig.patches.append(arrow)
fig.patches.append(arrow1)
fig.patches.append(arrow2)
fig.patches.append(arrow3)
fig.patches.append(arrow4)

# used annotate here to assist in the orginazation and placement of figures
plt.annotate(
    r"np.fft.fftshift($Ground$ $Cube$)",
    (0.43, 0.32),
    xycoords=fig.transFigure,
    va="center",
    weight="bold",
)

plt.tight_layout(h_pad=-1.8, w_pad=-2.5)

fig.text(
    0.65, 
    0.77, 
    r"np.fft.fftshift($Flip$ $Cube$)", 
    weight="bold",
)
fig.text(
    0.31, 
    0.77, 
    r"np.flip($\alpha$)", 
    weight="bold",
)

fig.text(
    0.48, 
    0.46,
    r"np.fft.fft2($Packed$ $Cube$)",
    weight="bold",
    rotation=17,
)
fig.text(
    0.48, 
    0.38,
    r"np.fft.ifft2($Packed$ $Cube$)",
    weight="bold",
    rotation=17,
)

fig.set_size_inches(13, 8)
plt.savefig(args.outfile, dpi=300)
