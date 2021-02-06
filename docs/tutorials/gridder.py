# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + nbsphinx="hidden"
# %matplotlib inline

# + nbsphinx="hidden"
# %run notebook_setup
# -

# # Gridding visibilities and making dirty images
#
# This tutorial covers how to read visibility data, average it to a "grid", and make diagnostic images.
# We'll start by using the mock CASA measurement set that we produced in the [visread](https://visread.readthedocs.io/en/latest/) [quickstart](https://visread.readthedocs.io/en/latest/tutorials/plot_baselines.html). You can create your own measurement set or download that one directly from [Zenodo](https://zenodo.org/record/4460128#.YB4ehGRKidZ).

import matplotlib.pyplot as plt
import numpy as np
import requests
import os

# For the purposes of autoexecuting this tutorial without inheriting the Python 3.6 dependency from CASA , we'll start directly from a saved NPZ file.

url = "https://zenodo.org/record/4498439/files/logo_cube.npz"
r = requests.get(url)
fname = "viz.npz"
with open(fname, "wb") as f:
    f.write(r.content)

d = np.load(fname)
uu = d["uu"]
vv = d["vv"]
weight = d["weight"]
data_re = d["data_re"]
data_im = -d[
    "data_im"
]  # we're converting from CASA convention to regular TMS convention

# But if you're fine working in Python 3.6, you could just as easily start by
#
#     import visread
#
#     vis = visread.read(filename="myMeasurementSet.ms")
#
#     vis.swap_convention(CASA_convention=False)
#     # access your data with
#     vis.frequencies  # frequencies in GHz
#     vis.uu  # East-West spatial frequencies in klambda
#     vis.vv  # North-South spatial frequencies in klambda
#     vis.weight # weight in 1/Jy^2
#     vis.data_re  # real components of visibilities in Jy
#     vis.data_im  # imaginary components of visibilities in Jy
#
# Be aware of the CASA convention to TSM convention, though.


import mpol
from mpol import gridding

# We'll start by creating a

gridder = gridding.Gridder(
    cell_size=0.005,  # [arcsec]
    npix=800,
    uu=uu,
    vv=vv,
    weight=weight,
    data_re=data_re,
    data_im=data_im,
)

# This has created a [GridCoords](https://mpol.readthedocs.io/en/latest/api.html#mpol.gridding.GridCoords) object attached to the [Gridder](https://mpol.readthedocs.io/en/latest/api.html#mpol.gridding.Gridder) object as `gridder.coords`. The choice of ``cell_size`` and ``npix`` define the size of the square image and Fourier grid.

gridder.grid_visibilities(weighting="uniform")
beam = gridder.get_dirty_beam()
img = gridder.get_dirty_image()

# There are also options to try out natural or Briggs' robust weighting. More information is in the [Gridder](https://mpol.readthedocs.io/en/latest/api.html#mpol.gridding.Gridder) API documentation.

# +
kw = {"origin": "lower", "interpolation": "none", "extent": gridder.coords.img_ext}

# the full dataset is 9 channels
chan = 4

gridder.grid_visibilities(weighting="uniform")
beam_uniform = gridder.get_dirty_beam()
img_uniform = gridder.get_dirty_image()

r = 0.5
gridder.grid_visibilities(weighting="briggs", robust=r)
beam_robust = gridder.get_dirty_beam()
img_robust = gridder.get_dirty_image()

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(6, 4.5))

ax[0, 0].imshow(beam_uniform[chan], **kw)
ax[0, 0].set_title("uniform")
ax[1, 0].imshow(img_uniform[chan], **kw)

ax[0, 1].imshow(beam_robust[chan], **kw)
ax[0, 1].set_title("robust={:}".format(r))
ax[1, 1].imshow(img_robust[chan], **kw)

fig.subplots_adjust(left=0.05, right=0.95, wspace=0.02, bottom=0.07, top=0.94)
# -

