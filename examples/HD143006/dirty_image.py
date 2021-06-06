from common_data import gridder
import matplotlib.pyplot as plt
import numpy as np

# Show the dirty image
img, beam = gridder.get_dirty_image(weighting="uniform", unit="Jy/arcsec^2")
kw = {"origin": "lower", "extent": gridder.coords.img_ext}
fig, ax = plt.subplots(ncols=1)
im = ax.imshow(np.squeeze(img), **kw)
plt.colorbar(im)
ax.set_title("image")
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
fig.savefig("uniform.png", dpi=300)

# Show the dirty image
img, beam = gridder.get_dirty_image(weighting="briggs", robust=1.0, unit="Jy/arcsec^2")
kw = {"origin": "lower", "extent": gridder.coords.img_ext}
fig, ax = plt.subplots(ncols=1)
im = ax.imshow(np.squeeze(img), **kw)
plt.colorbar(im)
ax.set_title("image")
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
fig.savefig("robust_1.0.png", dpi=300)

# Show the dirty image
img, beam = gridder.get_dirty_image(weighting="briggs", robust=0.0, unit="Jy/arcsec^2")
kw = {"origin": "lower", "extent": gridder.coords.img_ext}
fig, ax = plt.subplots(ncols=1)
im = ax.imshow(np.squeeze(img), **kw)
plt.colorbar(im)
ax.set_title("image")
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
fig.savefig("robust_0.png", dpi=300)

# Show the dirty image
img, beam = gridder.get_dirty_image(weighting="briggs", robust=-1.0, unit="Jy/arcsec^2")
kw = {"origin": "lower", "extent": gridder.coords.img_ext}
fig, ax = plt.subplots(ncols=1)
im = ax.imshow(np.squeeze(img), **kw)
plt.colorbar(im)
ax.set_title("image")
ax.set_xlabel(r"$\Delta \alpha \cos \delta$ [${}^{\prime\prime}$]")
ax.set_ylabel(r"$\Delta \delta$ [${}^{\prime\prime}$]")
fig.savefig("robust_-1.0.png", dpi=300)
