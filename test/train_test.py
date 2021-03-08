import numpy as np
import torch
import torch.optim
import matplotlib.pyplot as plt
from mpol import losses, precomposed
from mpol.constants import *


# configure a class to train with

# currently segfaults on 3.9
# https://github.com/pytorch/pytorch/issues/50014
def test_init_train_class(coords, dataset):

    nchan = dataset.nchan
    rml = precomposed.SimpleNet(coords=coords, nchan=nchan, griddedDataset=dataset)

    model_visibilities = rml.forward()

    rml.zero_grad()

    # calculate a loss
    loss = losses.nll(model_visibilities, dataset.vis_indexed, dataset.weight_indexed)

    # calculate gradients of parameters
    loss.backward()

    print(rml.bcube.base_cube.grad)


def test_train_loop(coords, dataset_cont, tmp_path):

    # set everything up to run on a single channel

    nchan = 1
    rml = precomposed.SimpleNet(coords=coords, nchan=nchan, griddedDataset=dataset_cont)

    optimizer = torch.optim.SGD(rml.parameters(), lr=0.001)

    for i in range(300):
        rml.zero_grad()

        # get the predicted model
        model_visibilities = rml.forward()

        # calculate a loss
        loss = losses.nll(
            model_visibilities, dataset_cont.vis_indexed, dataset_cont.weight_indexed
        )

        # calculate gradients of parameters
        loss.backward()

        # update the model parameters
        optimizer.step()

    # let's see what one channel of the image looks like
    fig, ax = plt.subplots(nrows=1)
    ax.imshow(
        np.squeeze(rml.icube.cube.detach().numpy()),
        origin="lower",
        interpolation="none",
        extent=rml.icube.coords.img_ext,
    )
    fig.savefig(tmp_path / "trained.png", dpi=300)
    plt.close("all")
