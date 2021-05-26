from astropy.utils.console import color_print
import matplotlib
from mpol import coordinates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.utils.data import download_file
from matplotlib.gridspec import GridSpec
from matplotlib import patches
from astropy.io import fits
import numpy as np

fname = download_file(
    "https://zenodo.org/record/4711811/files/logo_cont.fits?download=1",
    cache=True,
    show_progress=True,
    pkgname="mpol",
)
coords = coordinates.GridCoords(cell_size=2.1701388888889E-06 * 3600, npix=512)
kw = {"origin": "lower", "interpolation": "none", "extent": coords.img_ext}
kwvis = {"origin": "lower", "interpolation": "none", "extent": coords.vis_ext}

fig = plt.figure(constrained_layout=False)
gs = GridSpec(2, 3, fig)
ax = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,0],)
ax4 = fig.add_subplot(gs[1,2])



d = fits.open(fname)
print(d[0].header)
sky_cube = d[0].data
d.close()
ax.imshow(sky_cube, **kw)
ax.set_title('Sky Cube')
ax.set_xlabel(r'$\alpha$ [deg]')
ax.set_ylabel(r'$\delta$ [deg]')

flip_cube = np.flip(sky_cube, (1,))
ax1.imshow(sky_cube, **kw)
ax1.set_title('Flip Cube')
ax1.set_xlabel(r'$\alpha$ [deg]')
ax1.set_ylabel(r'$\delta$ [deg]')
ax1.invert_xaxis()

image_cube = np.fft.fftshift(flip_cube, axes=(0,1,))
ax2.imshow(image_cube, **kw)
ax2.set_title('Packed Cube (Image)')
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)

visibility_cube = np.fft.fft2(image_cube, axes=(0,1,))
ax3.imshow(np.abs(visibility_cube), norm=LogNorm(vmin=1e-4, vmax=12000), **kwvis)
ax3.set_title('Packed Cube (Visibility)')
ax3.xaxis.set_visible(False)
ax3.yaxis.set_visible(False)

ground_cube = np.fft.fftshift(visibility_cube, axes=(0,1,))
ax4.imshow(np.abs(ground_cube), **kwvis, norm=LogNorm(vmin=1e-4, vmax=12000))
ax4.set_title('Ground Cube')
ax4.set_xlabel(r'$u$ [k$\lambda$]')
ax4.set_ylabel(r'$v$ [k$\lambda$]')


arrow = patches.FancyArrowPatch((.65,.4),(.35,.4), transform=fig.transFigure, arrowstyle='->', mutation_scale=30, color='r')
arrow1 = patches.FancyArrowPatch((.65,.75),(.75,.75), transform=fig.transFigure, arrowstyle='->', mutation_scale=30, color='r')
arrow2 = patches.FancyArrowPatch((.31,.75),(.39,.75), transform=fig.transFigure, arrowstyle='->', mutation_scale=30, color='orange')
arrow3 = patches.FancyArrowPatch((.75,.57),(.32,.46), transform=fig.transFigure, arrowstyle='->', mutation_scale=30, color='green')
arrow4 = patches.FancyArrowPatch((.32,.43),(.75,.54), transform=fig.transFigure, arrowstyle='->', mutation_scale=30, color='blue')


fig.patches.append(arrow)
fig.patches.append(arrow1)
fig.patches.append(arrow2)
fig.patches.append(arrow3)
fig.patches.append(arrow4)


plt.annotate(r'torch.fft.fftshift($Packed$ $Cube$)', (.43,.37), xycoords=fig.transFigure, va='center', weight='bold', c='white')
# fig.subplots_adjust(left=.1, right=.9, bottom=0, top=.9)
plt.tight_layout(h_pad=-1.8, w_pad=-5)
fig.text(.4,.3,r'Red arrows = np.fft.fftshift($Packed$ $Cube$)', color='r', weight='bold')
fig.text(.4,.27,r'Orange arrow = np.flip($\alpha$)', color='orange', weight='bold')
fig.text(.4,.24,r'Green arrow = np.fft.fft2($Packed$ $Cube$)', color='green', weight='bold')
fig.text(.4,.21,r'Blue arrow = np.fft.ifft2($Packed$ $Cube$)', color='blue', weight='bold')

fig.set_size_inches(13,8)
plt.savefig('plot.png')
plt.show()