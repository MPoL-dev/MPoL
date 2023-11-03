import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.utils.data import download_file
from ray import tune

from mpol import (
    connectors,
    coordinates,
    crossval,
    datasets,
    gridding,
    images,
    losses,
    precomposed,
    utils,
)

# load the data
fname = "HD143006_continuum.npz"

d = np.load(fname)
uu = d["uu"]
vv = d["vv"]
weight = d["weight"]
data = d["data"]
data_re = data.real
data_im = data.imag

coords = coordinates.GridCoords(cell_size=0.01, npix=512)

gridder = gridding.Gridder(
    coords=coords,
    weight=weight,
    data_re=data_re,
    data_im=data_im,
)

dataset = gridder.to_pytorch_dataset()

# plot the grid
fig, ax = plt.subplots(nrows=1)
ax.imshow(
    np.squeeze(utils.packed_cube_to_ground_cube(dataset.mask).detach().numpy()),
    interpolation="none",
    origin="lower",
    extent=coords.vis_ext,
    cmap="GnBu",
)
fig.savefig("grid.png", dpi=300)


# create the cross validator
# create a radial and azimuthal partition
dartboard = datasets.Dartboard(coords=coords)

# create cross validator using this "dartboard"
k = 5
cv = crossval.DartboardSplitGridded(dataset, k, dartboard=dartboard, seed=42)
k_fold_datasets = [(train, test) for (train, test) in cv]

# create the model
model = precomposed.SimpleNet(coords=coords, nchan=dataset.nchan)

# create the residual connector
residuals = connectors.GriddedResidualConnector(model.fcube, dataset)
